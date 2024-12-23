from asyncore import write
import imp
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from torch.nn import CosineSimilarity
from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables, consistency_loss


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/hyh/yzj/SCCS-0.9027/SCCS/data_split/LA',
                    help='Name of Dataset')
parser.add_argument('--exp', type=str, default='SCCS', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int, default=2000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int, default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')  # 4
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')  # 8
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--gpu', type=str, default='1', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--magnitude', type=float, default='10.0', help='magnitude')
# -- setting of SCCS
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2 / 3, help='ratio of mask/image')
parser.add_argument('--tau', type=float, default=1, help='temperature of the contrastive loss')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()

DICE = losses.mask_DiceLoss(nclass=2)


def get_cut_mask(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks


def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)

    return torch.Tensor(batch_list).cuda()


def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


# def drop_loss(output, dropoutput):
#     BCE = nn.BCELoss()
#     # output_soft = F.softmax(output, dim=1)
#     pred = output[:, 0, :, :, :]
#     pred = torch.sigmoid(pred)
#     dropoutput_soft = F.softmax(dropoutput, dim=1)
#     lab = torch.argmax(dropoutput_soft, dim=1)
#     lab_int = lab.type(torch.int64)
#     lab_float = lab.float()
#     loss_dice = DICE(output, lab_int)
#     loss_ce = BCE(pred, lab_float)
#     return loss_dice, loss_ce

def drop_loss(output, dropoutput):
    """
    Consistency regularization between two augmented views
    """
    # 将特征图展平，除了批次维度，其他维度合并成一个向量
    # 此处我们假设批次是第一个维度，剩余的所有维度都被合并
    f_x_flat = output.view(output.size(0), -1)  # 改变形状为 (2, 2*112*112*80)
    f_x_delta_flat = dropoutput.view(dropoutput.size(0), -1)

    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(f_x_flat, f_x_delta_flat, dim=1)

    # 计算余弦距离
    cosine_distance = 1 - cosine_similarity

    # 计算损失值，通常我们对一个批次的余弦距离取平均作为最终的损失
    loss = cosine_distance.mean()
    return loss


def uncertainty_loss(inputs, targets, key_list):
    """
    Uncertainty rectified pseudo supervised loss
    """
    # detach from the computational graph
    uncertainty_loss = 0.0
    pseudo_label = F.softmax(targets / 0.5, dim=1).detach()
    keys = key_list if key_list is not None else list(inputs.keys())
    for key in keys:
        # pred = inputs[key][:, 0, :, :, :]
        # pred = torch.sigmoid(pred)
        vanilla_loss = F.cross_entropy(inputs[key], pseudo_label, reduction='none')
        # uncertainty rectification
        kl_div = torch.sum(
            F.kl_div(F.log_softmax(inputs[key], dim=1), F.softmax(targets, dim=1).detach(), reduction='none'),
            dim=1)
        uncertainty_loss += (torch.exp(-kl_div) * vanilla_loss).mean() + kl_div.mean()
    return uncertainty_loss


def BCE_loss(pred, target):
    pred = pred[:, 0, :, :, :]
    pred = torch.sigmoid(pred)
    # pred = torch.softmax(pred)
    target = target.float()
    # target = target.repeat(1,2,1,1,1)
    loss = nn.BCELoss()
    bce_loss = loss(pred, target)
    return bce_loss


def mix_volumes(img1, img2, labels1, labels2):
    # 生成并集掩码
    union_mask = torch.logical_or(labels1, labels2).int()

    # 初始化存储交换后图像和标签的张量
    swapped_img1 = torch.zeros_like(img1)
    swapped_img2 = torch.zeros_like(img2)
    swapped_labels1 = torch.zeros_like(labels1)
    swapped_labels2 = torch.zeros_like(labels2)

    # 对每个批次内的图像和标签利用并集掩码进行前背景交换
    for i in range(img1.size(0)):  # 遍历批次
        img1_single = img1[i]
        img2_single = img2[i]
        union_mask_single = union_mask[i]
        label1_single = labels1[i]
        label2_single = labels2[i]

        img1_foreground = img1_single * union_mask_single
        img1_background = img1_single * (1 - union_mask_single)

        img2_foreground = img2_single * union_mask_single
        img2_background = img2_single * (1 - union_mask_single)

        label1_foreground = label1_single * union_mask_single
        label1_background = label1_single * (1 - union_mask_single)

        label2_foreground = label2_single * union_mask_single
        label2_background = label2_single * (1 - union_mask_single)

        # 交换前景和背景
        swapped_img1[i] = img1_foreground + img2_background
        swapped_img2[i] = img2_foreground + img1_background

        swapped_labels1[i] = label1_foreground + label2_background
        swapped_labels2[i] = label2_foreground + label1_background

    return swapped_img1, swapped_img2, swapped_labels1, swapped_labels2


def mix_label(labelA, labelB, mask):
    mask = mask > 0
    mixed_label = labelA.clone()
    mixed_label[mask] = labelB[mask]

    return mixed_label


train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='mean')
BCE = nn.BCELoss()

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2


def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train", dropout=True)
    # model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    # sub_bs = int(args.labeled_bs / 4)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][
                                                                                  :args.labeled_bs]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:]
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            with torch.no_grad():
                # img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
                # mask_a = (lab_a > 0).float()
                # mask_b = (lab_b > 0).float()
                union_mask = torch.logical_or(lab_a, lab_b).int()
                # union_mask = union_mask.unsqueeze(dim=1)

            """Mix Input"""
            # input_a = mix_volumes(img_b, img_a, union_mask)
            # input_b = mix_volumes(img_a, img_b, union_mask)
            # mixlab_a = mix_label(lab_b, lab_a, union_mask)
            # mixlab_b = mix_label(lab_a, lab_b, union_mask)
            input_a, input_b, mixlab_a, mixlab_b = mix_volumes(img_a, img_b, lab_a, lab_b)

            out_a, _ = model(img_a)
            out_b, _ = model(img_b)
            output_a, _ = model(input_a)
            output_b, _ = model(input_b)

            loss_dice_a = DICE(output_a['output'], mixlab_a)
            loss_dice_b = DICE(output_b['output'], mixlab_b)
            loss_ce_a = BCE_loss(output_a['output'], mixlab_a)
            loss_ce_b = BCE_loss(output_b['output'], mixlab_b)
            loss_dice = loss_dice_a + loss_dice_b
            loss_ce = loss_ce_a + loss_ce_b



            loss = (loss_dice + loss_ce) / 4

            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info(
                'iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f' % (iter_num, loss, loss_dice, loss_ce))

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path, 'v9_iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model_v9.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    # torch.save(model.state_dict(), save_mode_path)
                    # torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def uncertainty_map(results):
    # mean = torch.mean(results,dim = 0)
    variance = torch.var(results,dim = 0)
    confidence_map = 1-variance
    weight_map = F.softmax(confidence_map).unsqueeze(0)
    weight_results = results * weight_map
    return torch.sum(weight_results, dim=0)


def self_train(args, pre_snapshot_path, self_snapshot_path):
    drop_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train", dropout=False)
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train", dropout=True)
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train", dropout=True)
    for param in ema_model.parameters():
        param.detach_()  # ema_model set
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
    labelnum = args.labelnum

    DICE = losses.mask_DiceLoss(nclass=2)
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    key_list = ['level3', 'level2', 'level1', 'output']
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    sub_bs = int(args.labeled_bs / 2)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model_v8.pth')
    load_net(model, pretrained_model)
    load_net(drop_model, pretrained_model)
    load_net(ema_model, pretrained_model)
    # contrastive_losses.cuda()


    model.train()
    drop_model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} itertations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs + sub_bs], volume_batch[
                                                                                       args.labeled_bs + sub_bs:]

            with torch.no_grad():
                # for _ in range(10):
                unoutput_ua, _ = ema_model(unimg_a)
                unoutput_ub, _ = ema_model(unimg_b)

                plab_a = get_cut_mask(unoutput_ua['output'], nms=1)
                plab_b = get_cut_mask(unoutput_ua['output'], nms=1)
                union_mask = torch.logical_or(plab_a, plab_a).int()
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # net_input_ua = mix_volumes(unimg_b, unimg_a, union_mask)
            # net_input_ub = mix_volumes(unimg_a, unimg_b, union_mask)
            # mixlab_a = mix_label(plab_b, plab_a, union_mask)
            # mixlab_b = mix_label(plab_a, plab_b, union_mask)
            net_input_ua, net_input_ub, mixlab_ua, mixlab_ub = mix_volumes(unimg_a, unimg_b, plab_a, plab_b)
            net_input_a, net_input_b, mixlab_a, mixlab_b = mix_volumes(img_a, img_b, lab_a, lab_b)

            outu_ua, _ = model(unimg_a)
            outu_ub, _ = model(unimg_b)
            out_ua, _ = model(net_input_ua)
            out_ub, _ = model(net_input_ub)
            drop_ua, _ = drop_model(net_input_ua)
            drop_ub, _ = drop_model(net_input_ub)

            out_a, _ = model(img_a)
            out_b, _ = model(img_b)

            ua_dice = DICE(out_ua['output'], mixlab_ua)
            ua_ce = BCE_loss(out_ua['output'], mixlab_ua)
            ub_dice = DICE(out_ub['output'], mixlab_ub)
            ub_ce = BCE_loss(out_ub['output'], mixlab_ub)

            la_dice = DICE(out_a['output'], lab_a)
            la_ce = BCE_loss(out_a['output'], lab_a)
            lb_dice = DICE(out_b['output'], lab_b)
            lb_ce = BCE_loss(out_b['output'], lab_b)

            drop_loss_ua = drop_loss(drop_ua['output'], out_ua['output'])
            drop_loss_ub = drop_loss(drop_ub['output'], out_ub['output'])

            l_drop = (drop_loss_ua + drop_loss_ub) / 2
            l_sup = (la_dice + la_ce + lb_dice + lb_ce) / 4

            l_semi_sup = (ua_dice + ua_ce + ub_dice + ub_ce) / 4
            # l_uncer = (uncer_loss_ua + uncer_loss_ub) / 2
            # l_semi_sup = (uncer_loss_ua + uncer_loss_ub) / 2


            loss = l_semi_sup + l_sup * 2 + l_drop * consistency_weight

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            # writer.add_scalar('Self/loss_l', loss_l, iter_num)
            # writer.add_scalar('Self/loss_u', loss_u, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logging.info('iteration %d : loss: %03f, loss_drop: %03f, loss_sup: %03f, loss_semi_sup: %03f' % (
            # iter_num, loss, l_drop, l_sup, l_semi_sup))
            # logging.info('iteration %d : loss: %03f, loss_contra: %03f, loss_sup: %03f, loss_semi_sup: %03f' % (
            # iter_num, loss, l_contra, l_sup, l_semi_sup))
            logging.info('iteration %d : loss: %03f, loss_sup: %03f, loss_semi_sup: %03f' % (
            iter_num, loss, l_sup, l_semi_sup))

            update_ema_variables(model, ema_model, 0.99)

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,
                                                  'v9_iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path, '{}_best_model_v9.pth'.format(args.model))
                    # save_net_opt(model, optimizer, save_mode_path)
                    # save_net_opt(model, optimizer, save_best_path)
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num % 200 == 1:
                ins_width = 2
                B, C, H, W, D = out_ua['output'].size()
                snapshot_img = torch.zeros(size=(D, 3, 3 * H + 3 * ins_width, W + ins_width), dtype=torch.float32)

                snapshot_img[:, :, H:H + ins_width, :] = 1
                snapshot_img[:, :, 2 * H + ins_width:2 * H + 2 * ins_width, :] = 1
                snapshot_img[:, :, 3 * H + 2 * ins_width:3 * H + 3 * ins_width, :] = 1
                snapshot_img[:, :, :, W:W + ins_width] = 1

                outputs_l_soft = F.softmax(out_ua['output'], dim=1)
                seg_out = outputs_l_soft[0, 1, ...].permute(2, 0, 1)  # y
                target = lab_a[0, ...].permute(2, 0, 1)
                train_img = net_input_ua[0, 0, ...].permute(2, 0, 1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_labeled' % (epoch, iter_num), snapshot_img)

                # outputs_u_soft = F.softmax(out_ub, dim=1)
                # seg_out = outputs_u_soft[0,1,...].permute(2,0,1) # y
                # target =  lab_b[0,...].permute(2,0,1)
                # train_img = net_input_ub[0,0,...].permute(2,0,1)

                snapshot_img[:, 0, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 1, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))
                snapshot_img[:, 2, :H, :W] = (train_img - torch.min(train_img)) / (
                            torch.max(train_img) - torch.min(train_img))

                snapshot_img[:, 0, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 1, H + ins_width:2 * H + ins_width, :W] = target
                snapshot_img[:, 2, H + ins_width:2 * H + ins_width, :W] = target

                snapshot_img[:, 0, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 1, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out
                snapshot_img[:, 2, 2 * H + 2 * ins_width:3 * H + 2 * ins_width, :W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_unlabel' % (epoch, iter_num), snapshot_img)

            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "./model/SCCS/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/SCCS/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Strating SCCS training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('../code/LA_train_SCCS.py', self_snapshot_path)
    # -- Pre-Training
    # logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)


