import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt
from medpy import metric
from monai.metrics.utils import distance_transform_edt
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
from scipy.spatial.distance import directed_hausdorff

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/hyh/yzj/SCCS-yzj/SCCS/data_split/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='SCCS', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, asd, hd95
    # return dice, hd95

def dice_coefficient(prediction, reference):
    intersection = np.sum(prediction * reference)
    size_prediction = np.sum(prediction)
    size_reference = np.sum(reference)

    try:
        dice = 2.0 * intersection / float(size_prediction + size_reference)
    except ZeroDivisionError:
        dice = 0.0

    return dice

def surface_distances(mask1, mask2):
    distance1 = distance_transform_edt(mask1, sampling=[1, 1])
    distance2 = distance_transform_edt(mask2, sampling=[1, 1])
    surface_distance = np.abs(distance1 - distance2)
    return surface_distance

def hausdorff_distance_95(mask1, mask2):
    surface_distance1 = surface_distances(mask1, mask2)
    surface_distance2 = surface_distances(mask2, mask1)
    # 检查数组是否为空或是否包含非零元素
    if np.any(surface_distance1 != 0) and np.any(surface_distance2 != 0):
        # 仅选择不等于 0 的数进行 concatenate
        surface_distances_combined = np.concatenate([surface_distance1.ravel()[surface_distance1.ravel() != 0],
                                                     surface_distance2.ravel()[surface_distance2.ravel() != 0]])
        hd95 = np.percentile(surface_distances_combined, 95)
        return hd95
    else:
        print("One or both of the arrays are empty or contain only zeros.")
        surface_distances_combined = np.array([])  # 或者根据你的需求设置一个默认值
        return 100


def save_tensor_as_image(input, lab, pred, folder_path, case, ind):
    # 1. 创建一个文件夹来保存图像
    if np.sum(lab == 1)==0 or np.sum(pred == 1)==0:
        dice1 = 0
        hd951 = 0
    else:
        dice1 = dice_coefficient(pred == 1, lab == 1)
        hd951 = hausdorff_distance_95(pred == 1, lab == 1)

    if np.sum(lab == 2)==0 or np.sum(pred == 2)==0:
        dice2 = 0
        hd952 = 0
    else:
        dice2 = dice_coefficient(pred == 2, lab == 2)
        hd952 = hausdorff_distance_95(pred == 2, lab == 2)

    if np.sum(lab == 3)==0 or np.sum(pred == 3)==0:
        dice3 = 0
        hd953 = 0
    else:
        dice3 = dice_coefficient(pred == 3, lab == 3)
        hd953 = hausdorff_distance_95(pred == 3, lab == 3)

    data_folder = os.path.join(folder_path, "{}".format(case))
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)

    plt.figure(figsize=(15, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(input, cmap="gray")
    plt.title('img_{}.png'.format(ind))

    plt.subplot(1, 3,  2)
    plt.imshow(lab, cmap="gray")
    plt.title('label_{}.png'.format(ind))

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title('pred_{}.png'.format(ind))


    plt.tight_layout()
    plt.figtext(0.5, 0.95, f'Dice1: {dice1:.2f}, HD951: {hd951:.2f}', ha='center', va='center', fontsize=12)
    plt.figtext(0.5, 0.92, f'Dice2: {dice2:.2f}, HD952: {hd952:.2f}', ha='center', va='center', fontsize=12)
    plt.figtext(0.5, 0.89, f'Dice3: {dice3:.2f}, HD953: {hd953:.2f}', ha='center', va='center', fontsize=12)
    plt.savefig(os.path.join(data_folder, "{}_{}".format(case, ind)))
    plt.close()

def save_test_single_volume(case, net, save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        show_label = label[ind, :, :]
        show_label = zoom(show_label, (256 / x, 256 / y), order=0)
        net.eval()
        with torch.no_grad():
            out_main = net(input)['output']
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            save_tensor_as_image(slice, show_label, out, save_path, case, ind)
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric

def test_single_volume(case, net, save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        show_label = label[ind, :, :]
        show_label = zoom(show_label, (256 / x, 256 / y), order=0)
        net.eval()
        with torch.no_grad():
            out_main = net(input)['output']
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()

            # save_tensor_as_image(slice, show_label, out, save_path, case, ind)
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric



def draw_line(x_data, y_data, x_label, y_label, title, rotation=0):
    plt.plot(x_data, y_data, marker='o',mec='r',mfc='w', markersize=3)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=-90)

    plt.xticks(range(len(x_data)), x_data)    # 在每个数据点上显示纵坐标的数值
    for x, y in zip(x_data, y_data):
        plt.text(x, y, f'{y:.2f}', ha='left', va='bottom', rotation= rotation)


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "./model/SCCS/ACDC_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
    test_save_path = "./model/SCCS/ACDC_{}_{}_labeled/{}_predictions_v16/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    # save_model_path = os.path.join(snapshot_path, '{}_best_model_v16.pth'.format(FLAGS.model))
    save_model_path = os.path.join('/home/hyh/yzj/SCCS-yzj/SCCS/models/ACDC/unet_best_model.pth')
    net.load_state_dict(torch.load(save_model_path))

    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    x_data = []
    y_data = []
    z_data = []
    for case in tqdm(image_list):
        # if case=="patient065_frame01" or case == "patient065_frame02" or case == "patient033_frame02" or case == "patient022_frame01" \
        #       or case == "patient022_frame02"or case == "patient059_frame01"or case == "patient059_frame02"or case == "patient065_frame02"or case == "patient083_frame02":
        #     first_metric, second_metric, third_metric = save_test_single_volume(case, net, test_save_path, FLAGS)
            # print("first_metric[3]",first_metric[3])
            # print("second_metric[3]", second_metric[3])
            # print("third_metric[3]", third_metric[3])
        # else:
        first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        avg_per_patient_dice = (first_metric[0] + second_metric[0] + third_metric[0]) / 3
        avg_per_patient_hd95 = (first_metric[3] + second_metric[3] + third_metric[3]) / 3
        # print("{}_dice:".format(case), avg_per_patient_dice)
        # print("{}_hd95:".format(case), avg_per_patient_hd95)
        # 这里把"{}_dice:".format(case)名字当横坐标，数值当纵坐标 ok
        x_data.append(str(case).replace('patient', '').replace('_frame', '_'))
        y_data.append(avg_per_patient_dice)
        z_data.append(avg_per_patient_hd95)

        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]

    y_data = [y * 10 for y in y_data]

    # 画折线图
    draw_line(x_data, y_data, "case", "dice/hd95", "case_dice", 45)
    draw_line(x_data, z_data, "case", "dice/hd95", "case_hd95", 45)
    plt.show()

    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    with open(test_save_path+'../performance_best.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))
