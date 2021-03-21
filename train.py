import argparse
import logging
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.distributed
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.ResNetEmbed import ResNetEmbed
from utils.util import  *
from utils.dataset import CubDataset

data_dir='./data/CUB_200_2011/images'
test_list='data/CUB_200_2011/test_images_4_level_V1.txt'
train_list='data/CUB_200_2011/train_images_4_level_V1.txt'

transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.GaussianBlur(),
        # transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )}


classes_dict = {'order': 13, 'family': 37, 'genus': 122, 'class': 200}


def train(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.d
    lr, batch_size, epochs, train_level = args.lr, args.bs, args.e, args.level

    # print("==> Reading the data ...")
    # test_set = CubDataset(data_dir, test_list, transforms['test'])
    # train_set = CubDataset(data_dir, train_list, transforms['train'])
    # test_num = len(test_set)
    # train_num = len(train_set)
    # tbs = 128 if opt.d == '0,1' else 64
    # test_loader = data.DataLoader(test_set,tbs,num_workers=8,pin_memory=True)
    # train_loader = data.DataLoader(train_set,batch_size,num_workers=8,pin_memory=True)
    # print("<== Data readed successfully.\n")

    print("==> Loading the network ...")
    model = ResNetEmbed(cdict=classes_dict)
    print("<== Network loaded successfully.\n")

    if args.w != '':
        print(f"==> Importing network weights from {args.w} ...")
        model.load_state_dict(torch.load(args.w))
        print("<== Network weights imported successfully.\n")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"-{name}-: {param.requires_grad}")

    # model = model.cuda(opt.d)


    # optimizer = optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.00005)
    # loss_function = nn.CrossEntropyLoss()
    # hierarchyUtils = dataset.HierarchyUtil()
    #
    # train_acc_list = []
    # test_acc_list = []
    # train_loss_list = []
    # test_loss_list = []
    #
    # epoch_list = [epoch + 1 for epoch in range(epochs)]
    # for epoch in range(1, epochs + 1):
    #     net.train()
    #     train_bar = tqdm(train_loader)
    #     train_bar.desc = f"train epoch[{epoch}/{epochs}]"
    #     for featrues, labels in train_bar:
    #         featrues = featrues.cuda()
    #         labels = labels.cuda()
    #         optimizer.zero_grad()
    #         results = net(featrues)
    #         labels = hierarchyUtils.get_hierarchy(labels)
    #         loss = loss_function(results[train_level], labels[train_level])
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #     net.val()
    #     total_loss = [0.0, 0.0]
    #     acc = [0.0, 0.0]
    #     with torch.no_grad():
    #         bar = tqdm(test_train_loader)
    #         bar.desc = train_bar.desc = f"Test train data epoch[{epoch}/{epochs}]"
    #         for featrues, labels in bar:
    #             featrues = featrues.cuda()
    #             labels = labels.cuda()
    #             results = net(featrues)[train_level]
    #             labels = hierarchyUtils.get_hierarchy(labels)[train_level]
    #             val_loss = loss_function(results, labels)
    #             total_loss[0] += val_loss.item() / num_data[0]
    #             y_hat = torch.max(results, dim=1)[1]
    #             acc[0] += torch.eq(y_hat, labels).sum().item()
    #
    #         bar = tqdm(test_loader)
    #         bar.desc = train_bar.desc = f"Test test data epoch[{epoch}/{epochs}]"
    #         for featrues, labels in bar:
    #             featrues = featrues.cuda()
    #             labels = labels.cuda()
    #             results = net(featrues)[train_level]
    #             labels = hierarchyUtils.get_hierarchy(labels)[train_level]
    #             val_loss = loss_function(results, labels)
    #             total_loss[1] += val_loss.item() / num_data[1]
    #             y_hat = torch.max(results, dim=1)[1]
    #             acc[1] += torch.eq(y_hat, labels).sum().item()
    #
    #     print(f"train Loss: {round(total_loss[0], 5)}  train Acc: {round(acc[0], 5)}"
    #           f"test  Loss: {round(total_loss[1], 5)}  test  Acc: {round(acc[0], 5)}")
    #
    #     train_acc_list.append(acc[0])
    #     test_acc_list.append(acc[1])
    #     train_loss_list.append(total_loss[0])
    #     test_loss_list.append(total_loss[1])
    #
    #     plt.figure(figsize=(20, 10))
    #     # 绘制验证正确率曲线图
    #     plt.plot(epoch_list, train_acc_list, color='cyan', label='train acc')
    #     plt.plot(epoch_list, test_acc_list, color='red', label='test acc')
    #     plt.legend()
    #     plt.xlabel('epochs')
    #     plt.ylabel('accuracy')
    #     plt.title(levelTupleList[train_level][0] + "accuracy")
    #     acc_list = [0.0, 0.5, 1.0, max(train_acc_list)]
    #     e_list = [i for i in range(0, epochs, 10)]
    #     e_list.append(epochs)
    #     plt.xticks(e_list)
    #     plt.yticks(acc_list)
    #     plt.axhline(y=max(train_acc_list), c="green", ls="--")
    #     plt.grid(True)
    #     plt.savefig(save_path + levelTupleList[train_level][0] + "accuracy.jpg")
    #     plt.clf()
    #
    #     # 绘制损失曲线图
    #     plt.plot(epoch_list, train_loss_list, color='cyan', label='train_loss')
    #     plt.plot(epoch_list, test_loss_list, color='red', label='test_loss')
    #     plt.legend()
    #     plt.xlabel('epochs')
    #     plt.ylabel('loss')
    #     plt.title('ResNet50 in CUB200')
    #     plt.xticks(e_list)
    #     plt.yticks([0.0, 0.05, 0.10, min(train_loss_list)])
    #     plt.grid(True)
    #     plt.savefig(save_path + levelTupleList[train_level][0] + 'trainloss.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', type=int, default=25,help="epochs")
    parser.add_argument('--bs', type=int, default=8,help="batch size")
    parser.add_argument('--d', type=str, default='0', help='cuda device')
    parser.add_argument('--w', type=str, default='', help='initial weights path')
    parser.add_argument('--nopretrain', action='store_true', help="don't pretrained resnet50")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--level', type=str, default='order', help='training level')
    parser.add_argument('--r', type=int, default=2)

    opt = parser.parse_args()

    # create a path to save the train result and model.
    # save_path = make_save_path(classes_dict[opt.level])
    # logger = make_logger(save_path,__name__)
    # logger.info("-------LOG--------")
    # logger.info(opt)
    # logger.info("------------------")
    # logger.info("save in " + save_path)
    # logger.info("------------------")

    train(opt)


