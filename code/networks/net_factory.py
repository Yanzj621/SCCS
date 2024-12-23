from SCCS.code.networks.unet import UNet, UNet_2d
from SCCS.code.networks.VNet import VNet
from SCCS.code.networks.unet_dropout import drop_UNet_2d
import torch.nn as nn

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0, dropout=False):
    if net_type == "unet" and mode == "train":
        # net = UNet(in_chns=in_chns, class_num=class_num ).cuda()
        net = UNet_2d(in_chns=in_chns, class_num=class_num, drop_out=False).cuda()
    if net_type == "VNet" and mode == "train" and tsne == 0 and dropout == False:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    if net_type == "VNet" and mode == "train" and tsne == 0 and dropout == True:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet" and mode == "test" and tsne == 0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net

# def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0):
#     if net_type == "unet" and mode == "train":
#         net = UNet(in_chns=in_chns, class_num=class_num).cuda()
#     if net_type == "VNet" and mode == "train" and tsne == 0:
#         net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
#     if net_type == "VNet" and mode == "test" and tsne == 0:
#         net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
#     return net

def BCP_net(in_chns=1, class_num=2, ema=False, drop=False):
    if drop == False:
        net = UNet_2d(in_chns=in_chns, class_num=class_num,drop_out=False).cuda()
    else:
        net = UNet_2d(in_chns=in_chns, class_num=class_num, drop_out=True).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

# def drop_BCP_net(in_chns=1, class_num=2, ema=False):
#     net = drop_UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
#     if ema:
#         for param in net.parameters():
#             param.detach_()
#     return net

