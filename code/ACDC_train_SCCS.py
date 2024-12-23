import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from torch.optim import Adam

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from SCCS.code.networks.net_factory import BCP_net
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d, transformdata, losses_v1
# from utils import transformsgpu
# from utils import transformmasks

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/hyh/yzj/SCCS-0.9027（2）/SCCS/data_split/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SCCS', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')# 24
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')# 12
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')#7
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--gamma', type=int, default=0.6, help='')
# parser.add_argument('--color_jitter', tpre=bool, default=True, help='color_jitter')
# parser.add_argument('--gaussian_blur', type=bool, default=True, help='GaussianBlur')
# costs
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')
parser.add_argument('--consistency_type', type=str,  default="kl", help='consistency_type')

args = parser.parse_args()

dice_loss = losses_v1.DiceLoss(n_classes=4)
contrastive_losses = losses_v1.contra_loss(temperature=0.5, batch_size=2, num_class=4)

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)



def l_loss(output, lab):
    CE = nn.CrossEntropyLoss(reduction='mean')
    lab = lab.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    loss_dice = dice_loss(output_soft, lab.unsqueeze(1))
    loss_ce = CE(output, lab)

    return loss_dice, loss_ce

def drop_loss(output, dropoutput):

    f_x_flat = output.view(output.size(0), -1)
    f_x_delta_flat = dropoutput.view(dropoutput.size(0), -1)
    cosine_similarity = F.cosine_similarity(f_x_flat, f_x_delta_flat, dim=1)
    cosine_distance = 1 - cosine_similarity
    cos_loss = cosine_distance.mean()

    return cos_loss

#
def uncertainty_loss(inputs, targets, key_list):
    """
    Uncertainty rectified pseudo supervised loss
    """
    # detach from the computational graph
    uncertainty_loss = 0.0
    pseudo_label = F.softmax(targets / 0.5, dim=1).detach()
    keys = key_list if key_list is not None else list(inputs.keys())
    for key in keys:
        vanilla_loss = F.cross_entropy(inputs[key], pseudo_label, reduction='none')
        # uncertainty rectification
        kl_div = torch.sum(F.kl_div(F.log_softmax(inputs[key], dim=1), F.softmax(targets, dim=1).detach(), reduction='none'),
                       dim=1)
        uncertainty_loss  += (torch.exp(-kl_div) * vanilla_loss).mean() + kl_div.mean()
    return uncertainty_loss


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        # 字典类型：因为patiens_num=7,所以下面的labeled_slice=136
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def getWeakInverseTransformParameters(parameters):
    return parameters

#
# def show_images(images_tensor):
#     # 确保输入是一个四维的Tensor [batch_size, channels, height, width]
#     if len(images_tensor.shape) != 4:
#         # 设置绘图
#       fig, axes = plt.subplots(1, len(images_tensor), figsize=(15, 5))
#       for i, img in enumerate(images_tensor):
#           ax = axes[i] if len(images_tensor) > 1 else axes
#           # 显示图像
#           img = img.squeeze().cpu().numpy()  # 假设图像已经是0-1或0-255范围，去掉通道维度，并转换为numpy数组
#           ax.imshow(img, cmap='gray')  # 以灰度图形式显示
#           ax.axis('off')  # 不显示坐标轴
#       plt.show()
#     # 设置绘图
#     else:
#       fig, axes = plt.subplots(1, len(images_tensor), figsize=(15, 5))
#       for i, img in enumerate(images_tensor):
#           ax = axes[i] if len(images_tensor) > 1 else axes
#           # 图像Tensor从1x256x256转换为256x256
#           img = img.squeeze().cpu().numpy()  # 假设图像已经是0-1或0-255范围，去掉通道维度，并转换为numpy数组
#           ax.imshow(img, cmap='gray')  # 以灰度图形式显示
#           ax.axis('off')  # 不显示坐标轴
#       plt.show()
#
#
# def highlight_boundaries(image_tensor, mask1_tensor, mask2_tensor, color1=(255,0,0), color2=(0,0,255), thickness=2, alpha = 0.3):
#     # 确保Tensor在CPU上，并转换为numpy数组
#     if image_tensor.is_cuda:
#         image_tensor = image_tensor.cpu()
#     if mask1_tensor.is_cuda:
#         mask1_tensor = mask1_tensor.cpu()
#     if mask2_tensor.is_cuda:
#         mask2_tensor = mask2_tensor.cpu()
#
#     image = image_tensor.numpy().squeeze()
#     mask1 = mask1_tensor.numpy().squeeze()
#     mask2 = mask2_tensor.numpy().squeeze()
#     image = (image * 255).astype(np.uint8)
#     mask1 = (mask1 * 255).astype(np.uint8)
#     mask2 = (mask2 * 255).astype(np.uint8)
#
#     # 确保掩码是二进制的
#     _, thresholded_mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
#
#     # 找到掩码的边界
#     contours1, _ = cv2.findContours(thresholded_mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 在原图上画出边界
#     output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     cv2.drawContours(output_image, contours1, -1, color1, thickness)
#     # 创建一个相同大小的高亮色层
#     highlight_layer = np.zeros_like(output_image, np.uint8)
#     highlight_layer[:] = color2
#
#     # 创建一个蒙版，用于合并图像
#     mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)[1]
#     mask2 = mask2.astype(bool)
#
#     # 将高亮色层合并到原图
#     output_image[mask2] = cv2.addWeighted(output_image[mask2], 1 - alpha, highlight_layer[mask2], alpha, 0)
#
#     return output_image
#
#
# def process_and_show_images(images_tensor, masks1_tensor,masks2_tensor):
#     batch_size = images_tensor.size(0)
#
#     # 设置绘图
#
#
#     for i in range(batch_size):
#         image_tensor = images_tensor[i]
#         mask1_tensor = masks1_tensor[i]
#         mask2_tensor = masks2_tensor[i]
#
#         highlighted_image = highlight_boundaries(image_tensor, mask1_tensor, mask2_tensor)
#
#
#
#         # 显示带边界的图
#         # ax = axes[i, 1] if batch_size > 1 else axes[1]
#         plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.show()



def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model_v2.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)


    model = BCP_net(in_chns=1, class_num=num_classes,drop=False )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    # db_train加载的是train_slices.list里的数据
    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)#2

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=2)#2

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    weak_parameters = {"flip": 0}
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]

            mask_a = (lab_a > 0).float()
            mask_b = (lab_b > 0).float()

            union_mask = torch.logical_or(mask_a, mask_b).int()
            union_mask = union_mask.view(6, 1, 256, 256) #(6,1,256,256)
            loss_mask = union_mask.squeeze(dim=1)

            # 取a前景b背景
            input_a = img_a*union_mask+img_b*(1-union_mask)
            # 取b前景a背景
            input_b = img_b*union_mask+img_a*(1-union_mask)


            output_a = model(input_a)['output']
            output_b = model(input_b)['output']


            loss_dice_a, loss_ce_a = l_loss(output_a, lab_a)
            loss_dice_b, loss_ce_b = l_loss(output_b, lab_b)

            loss_dice = loss_dice_a + loss_dice_b
            loss_ce = loss_ce_a + loss_ce_b
            loss = (loss_dice + loss_ce) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            if iter_num % 20 == 0:

                image_a = input_a[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image_a', image_a, iter_num)
                outputs_a = torch.argmax(torch.softmax(output_a, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction_a', outputs_a[1, ...] * 50, iter_num)
                # labs_a = mixlab_a[1, ...].unsqueeze(0) * 50
                # writer.add_image('pre_train/Mixed_GroundTruth', labs_a, iter_num)

                image_b = input_b[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image_b', image_b, iter_num)
                outputs_b = torch.argmax(torch.softmax(output_b, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction_b', outputs_b[1, ...] * 50, iter_num)
                # labs_b = mixlab_b[1, ...].unsqueeze(0) * 50
                # writer.add_image('pre_train/Mixed_GroundTruth', labs_b, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,patch_size=[256,256])
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'best_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model_best.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    # 把最优模型的参数保存在字典里
                    # torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
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

def self_train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model_best.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)

    model = BCP_net(in_chns=1, class_num=num_classes,drop=False)
    drop_model = BCP_net(in_chns=1, class_num=num_classes,drop=True)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)
    key_list = ['level3','level2','level1','output']

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)#4

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)#1

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)


    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    load_net(drop_model, pre_trained_model)



    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()
    drop_model.train()


    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()


            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[
                                                                                               args.labeled_bs + unlabeled_sub_bs:]
            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            with torch.no_grad():
                # for _ in range(10):
                pre_a = ema_model(uimg_a)['output']
                pre_b = ema_model(uimg_b)['output']
                #     result_ua.append(pre_a)
                #     result_ub.append(pre_a)
                # re_ua = torch.stack(result_ua)
                # re_ub = torch.stack(result_ub)
                # uncertainty_mapua = uncertainty_map(re_ua)
                # uncertainty_mapub = uncertainty_map(re_ub)

                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                mask_ua = (plab_a > 0).float()
                mask_ub = (plab_b > 0).float()


                union_mask = torch.logical_or(mask_ua, mask_ub).int()
                union_mask = union_mask.view(6, 1, 256, 256) #(6,1,256,256)
                loss_mask = union_mask.squeeze(dim=1)



            # 前景是uimg_a,背景是img_a
            net_input_ua = uimg_a * union_mask + uimg_b * (1 - union_mask)
            net_input_ub = uimg_b * union_mask + uimg_a * (1 - union_mask)


            out_ua = model(net_input_ua)
            out_ub = model(net_input_ub)


            drop_ua = drop_model(net_input_ua)
            drop_ub = drop_model(net_input_ub)

            out_a = model(img_a)
            out_b = model(img_b)


            ua_dice, ua_ce = l_loss(out_ua['output'], plab_a)
            ub_dice, ub_ce = l_loss(out_ub['output'], plab_b)

            droploss_ua = drop_loss(out_ua['output'], drop_ua['output'])
            droploss_ub = drop_loss(out_ub['output'], drop_ub['output'])


            la_dice, la_ce = l_loss(out_a['output'], lab_a)
            lb_dice, lb_ce = l_loss(out_b['output'], lab_b)

            uncer_loss_ua = uncertainty_loss(out_ua, pre_a, key_list=key_list)
            uncer_loss_ub = uncertainty_loss(out_ub, pre_b, key_list=key_list)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss_ce = ua_ce + ub_ce
            loss_dice = ua_dice + ub_dice

            l_drop = (droploss_ua + droploss_ub ) / 2
            l_sup = (la_dice + la_ce + lb_dice + lb_ce) / 4
            # semi_supervised_loss = (loss_dice + loss_ce) / 2
            # uncer_loss = (uncer_loss_ua + uncer_loss_ub) / 2
            semi_supervised_loss = (uncer_loss_ua + uncer_loss_ub) / 2


            # loss = semi_supervised_loss + l_sup * 2 + l_drop * 0.2
            loss = semi_supervised_loss + l_sup * 2 + l_drop * 0.4
            # loss = semi_supervised_loss + l_sup * 2 + l_drop * consistency_weight


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            update_model_ema(model, ema_model, 0.99)
            # update_model_ema(model, drop_model, 0.99)

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_b', loss_ce, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
            # logging.info('iteration %d: loss: %f' % (iter_num, loss))

            if iter_num % 20 == 0:
                image_ua = uimg_a[1, 0:1, :, :]
                writer.add_image('train/Un_Image', image_ua, iter_num)
                outputs = torch.argmax(torch.softmax(pre_a, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                # labs = unl_label[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/Un_GroundTruth', labs, iter_num)

                # image_ub = uimg_b[1, 0:1, :, :]
                # writer.add_image('train/Un_Image', image_ub, iter_num)
                # outputs = torch.argmax(torch.softmax(pre_b, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
                # labs = l_label[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/Un_GroundTruth', labs, iter_num)

                # image_la = imga[1, 0:1, :, :]
                # writer.add_image('train/L_Image', image_la, iter_num)
                # outputs_l = torch.argmax(torch.softmax(out_la, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                # labs_l = l_label[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/L_GroundTruth', labs_l, iter_num)

                # image_lb = imgb[1, 0:1, :, :]
                # writer.add_image('train/L_Image', image_lb, iter_num)
                # outputs_l = torch.argmax(torch.softmax(out_lb, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
                # labs_l = l_label[1, ...].unsqueeze(0) * 50
                # writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes,patch_size=[256,256])
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'b=0.4_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model_b=0.4.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()
                # scheduler.step()
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "./model/SCCS/ACDC_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/SCCS/ACDC_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('ACDC_train_SCCS.py', self_snapshot_path)

    #Pre_train
    # logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    # pre_train(args, pre_snapshot_path)

    #Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    


