import argparse
import os
import copy
from loss import HuberLoss, CharbonnierLoss
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import ESPCN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str,default='BLAH_BLAH/General-191_x3.h5')#'BLAH_BLAH/91-image_x3.h5',BLAH_BLAH/General-191_x8.h5
    # parser.add_argument('--train-file1', type=str, default='BLAH_BLAH/91-image_x2.h5')  # 'BLAH_BLAH/91-image_x3.h5'
    parser.add_argument('--eval-file', type=str,default='BLAH_BLAH/Set5_x3.h5')#评估
    parser.add_argument('--eval-file1', type=str, default='BLAH_BLAH/Set14_x3.h5')  # 评估
    parser.add_argument('--eval-file2', type=str, default='BLAH_BLAH/BSD200_x3.h5')  # 评估
    parser.add_argument('--outputs-dir', type=str,default='BLAH_BLAH/outputs')
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))#输出文件地址BLAH_BLAH/outputs/x3

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
   #建立模型
    model = ESPCN(scale_factor=args.scale).to(device)
    criterion =CharbonnierLoss(delta=0.0001)#CharbonnierLoss(delta=0.0001)#HuberLoss(delta=0.9)#nn.L1Loss()# nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.first_part.parameters(), 'lr': args.lr * 0.1},
        # {'params': model.mid_part.parameters(), 'lr': args.lr * 0.1},
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    #数据集加载
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=False)#drop_last=False
    #############x2
    # train_dataset1 = TrainDataset(args.train_file1)
    # train_dataloader1 = DataLoader(dataset=train_dataset1,
    #                               batch_size=args.batch_size,
    #                               shuffle=True,
    #                               num_workers=args.num_workers,
    #                               pin_memory=True,
    #                               drop_last=False)  # drop_last=False
    #加载评估集
    #Set5
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    #Set14
    eval_dataset1 = EvalDataset(args.eval_file1)
    eval_dataloader1 = DataLoader(dataset=eval_dataset1, batch_size=1)
    #BSD200
    eval_dataset2 = EvalDataset(args.eval_file2)
    eval_dataloader2 = DataLoader(dataset=eval_dataset2, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    epoch_num=range(1,args.num_epochs+1)
    psrn=[]
    loss_num=[]
    psrn_Set14=[]
    psrn_BSD200=[]

    # 开始训练
    for epoch in range(args.num_epochs):
        # 更新lr，动态修改学习率，不同的迭代次数学习率的选择不同
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.1 ** (epoch // int(args.num_epochs * 0.8)))#//运算符只保留计算结果的整数部分
            print(param_group['lr'])


        model.train()
        epoch_losses = AverageMeter()
        # tqdm在 Python 长循环中添加一个进度提示信息
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()#把模型中参数的梯度设为0
                loss.backward()
                optimizer.step()#optimizer.step()也就是当前参数空间对应的梯度，因为如果不清零，那么使用的这个grad就得同上一个mini-batch有关

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        #存储每次迭代后的模型
        #torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        model.eval()
        epoch_psnr = AverageMeter()
        epoch_psnr1 = AverageMeter()
        epoch_psnr2 = AverageMeter()
        # 验证模型Set5
        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)  # 把数据限制在0~1

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        #Set14
        for data in eval_dataloader1:
            inputs1, labels1 = data

            inputs1 = inputs1.to(device)
            labels1 = labels1.to(device)

            with torch.no_grad():
                preds1 = model(inputs1).clamp(0.0, 1.0)  # 把数据限制在0~1

            epoch_psnr1.update(calc_psnr(preds1, labels1), len(inputs1))
        #BSD200
        for data in eval_dataloader2:
            inputs2, labels2 = data

            inputs2 = inputs2.to(device)
            labels2 = labels2.to(device)

            with torch.no_grad():
                preds2 = model(inputs2).clamp(0.0, 1.0)  # 把数据限制在0~1

            epoch_psnr2.update(calc_psnr(preds2, labels2), len(inputs2))

        # 计算平均PSNR值
        print('Set-5 eval psnr: {:.2f}'.format(epoch_psnr.avg))
        print('Set-14 eval psnr: {:.2f}'.format(epoch_psnr1.avg))
        print('BSD200 eval psnr: {:.2f}'.format(epoch_psnr2.avg))
        #绘制psrn
        psrn.append(epoch_psnr.avg)
        psrn_Set14.append(epoch_psnr1.avg)
        psrn_BSD200.append(epoch_psnr2.avg)
        loss_num.append(epoch_losses.avg)
        # 记录所有epochs中最好的PSNR模型
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    # 绘制PSRN曲线图
    plt.figure()
    plt.plot(epoch_num, psrn, label='Set5')  # label定义两条线的名称
    plt.plot(epoch_num, psrn_Set14, label='Set14')  # label定义两条线的名称
    plt.plot(epoch_num, psrn_BSD200, label='BSD200')  # label定义两条线的名称
    plt.legend()  # 添加图例

    # 数轴的描述
    plt.xlabel('Number of iterations')
    plt.ylabel('Average Test PSNR (dB)')
    plt.grid()
    # 改变坐标轴的位置
    ax = plt.gca()  # 获取图片
    ax.spines['right'].set_color('none')  # 将右边的线去除
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')  # 将下面的线设置为x轴
    ax.yaxis.set_ticks_position('left')
    #绘制loss函数
    plt.figure()
    plt.plot(epoch_num, loss_num, label='Loss')  # label定义两条线的名称
    plt.legend()  # 添加图例
    # 数轴的描述
    plt.xlabel('Number of iterations')
    plt.ylabel('Average Test Loss ')
    plt.grid()
    # 改变坐标轴的位置
    ax = plt.gca()  # 获取图片
    ax.spines['right'].set_color('none')  # 将右边的线去除
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')  # 将下面的线设置为x轴
    ax.yaxis.set_ticks_position('left')
    #Set5 和Set14
    plt.figure()
    plt.plot(epoch_num, psrn, label='PSNR-Set5')  # label定义两条线的名称
    plt.plot(epoch_num, psrn_Set14, label='PSNR-Set14')  # label定义两条线的名称
    plt.legend()  # 添加图例
    # 数轴的描述
    plt.xlabel('Number of iterations')
    plt.ylabel('Average Test PSNR (dB)')
    plt.grid()
    # 改变坐标轴的位置
    ax = plt.gca()  # 获取图片
    ax.spines['right'].set_color('none')  # 将右边的线去除
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')  # 将下面的线设置为x轴
    ax.yaxis.set_ticks_position('left')
    plt.show()

    #选择最好的best epoch
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

