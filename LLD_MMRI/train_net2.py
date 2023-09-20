import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
from net import net2
from tqdm import tqdm
from utils import ACC
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from data.dataset_net2 import LLD_dataset, RandomGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../LLD-MMRI2023/data', help='root dir for data')  # 根路径
parser.add_argument('--dataset', type=str,
                    default='LLD', help='experiment_name')  # 数据路径
parser.add_argument('--lld_dir', type=str,
                    default='./lld/lld_mmri', help='lld dir')  # 数据列表路径
parser.add_argument('--num_classes', type=int,
                    default=7, help='output channel of network')  # 分类数目
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')  # ？
parser.add_argument('--epochs', type=int,
                    default=500, help='maximum epoch number to train')  # ？
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')  # batch_size
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')  # gpu数量
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')  # ？
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=128, help='input patch size of network input')  # 数据尺寸
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')  # 随机种子？
parser.add_argument('--vit_name', type=str,
                    default='net2', help='select one vit model')  # 模型名称

args = parser.parse_args(args=[])


def trainer_lld(args, model, lld_path):
    logging.basicConfig(filename=lld_path + "/log_net2.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = LLD_dataset(base_dir=args.root_path, lld_dir=args.lld_dir, split="train_net2",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = LLD_dataset(base_dir=args.root_path, lld_dir=args.lld_dir, split="val_net2")
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True,
                           worker_init_fn=worker_init_fn)
    if args.n_gpu == 1:
        model = model.cuda()
    elif args.n_gpu == 2:
        model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    ce_loss = CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(lld_path + '/log')
    iter_num = 0
    epoch = args.epochs
    trainloader_len = len(trainloader)
    valloader_len = len(valloader)
    max_iterations = args.epochs * trainloader_len  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(trainloader_len, max_iterations))

    last_dice = 0.7

    for epoch_num in range(epoch):
        loss_sum = 0
        val_loss_sum = 0
        val_acc = 0
        print(epoch_num)
        model.train()
        for i_batch, sampled_batch in tqdm(enumerate(trainloader), total=trainloader_len):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)

            loss = ce_loss(outputs, label_batch)

            loss_sum = loss_sum + loss.item()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

        loss_avg = loss_sum / len(trainloader)
        writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], epoch_num)
        writer.add_scalar('info/total_loss', loss_avg, epoch_num)
        logging.info('iteration %d : loss : %f' % (epoch_num, loss_avg))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in tqdm(enumerate(valloader), total=valloader_len):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                #                             print(image_batch.shape)
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(image_batch)
                loss = ce_loss(outputs, label_batch)
                val_loss_sum = val_loss_sum + loss.item()
                acc = ACC(torch.softmax(outputs, dim=1), label_batch)
                val_acc = val_acc + acc
            val_loss_avg = val_loss_sum / len(valloader)
            acc_avg = val_acc / len(valloader)
            logging.info('val_log %d : loss : %f  acc : %f' % (epoch_num, val_loss_avg, acc_avg))
            if acc_avg > last_dice or (acc_avg > last_dice - 0.0 and epoch_num > 100):
                last_dice = max(acc_avg, last_dice)
                save_mode_path = os.path.join(lld_path, 'epoch_' + str(epoch_num) + '_acc_' + str(acc_avg) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    args.is_pretrain = True
    args.exp = dataset_name + str(args.img_size)
    lld_path = "model/" + args.vit_name

    if not os.path.exists(lld_path):
        os.makedirs(lld_path)

    net = net2.U_Net()

    trainer_lld(args, net, lld_path)
