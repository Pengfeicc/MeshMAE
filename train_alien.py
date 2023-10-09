import torch
import os
import sys
from torch.autograd import Variable
import argparse
from tensorboardX import SummaryWriter
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import random
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from model.dataset import SegmentationDataset
from model.meshmae import Mesh_baseline_seg
from model.reconstruction import save_results
from torchmetrics.classification import MulticlassJaccardIndex
import sys

sys.setrecursionlimit(3000)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(net, optim, criterion, train_dataset, epoch, args):
    net.train()
    running_loss = 0
    running_corrects = 0
    n_samples = 0
    patch_size = 64
    num_of_patch = 0
    for i, (face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels) in enumerate(
            train_dataset):
        optim.zero_grad()
        faces = face_patch.cuda()   #faces-> ([2, 256, 64, 3])
        patch_size = faces.size(2)   #patch_size-> 64
        num_of_patch = faces.size(1)   #num_of_patch-> 256

        feats = feats_patch.to(torch.float32).cuda()   #feats-> ([2, 256, 64, 10])
        centers = center_patch.to(torch.float32).cuda()   #centers-> ([2, 256, 64, 3])
        Fs = np_Fs.cuda()   #Fs-> ([2, 256, 64, 1])
        cordinates = coordinate_patch.to(torch.float32).cuda()  #cordinates-> ([2, 256, 64, 9])
        labels = labels.to(torch.long).cuda()   #labels-> ([2, 256, 64, 1])

        labels = labels.reshape(faces.shape[0], -1)   #labels-> ([2, 16384])
        n_samples += faces.shape[0]   #n_samples-> 2, 4, ...

        outputs, outputs_seg = net(faces, feats, centers, Fs, cordinates)
        outputs = outputs.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)

        dim = outputs.shape[1]

        loss = criterion(outputs, labels)
        loss_seg = criterion(outputs_seg, labels)
        loss = args.lw1 * loss + args.lw2 * loss_seg

        _, preds = torch.max(outputs_seg, 1)   #preds-> ([2, 16384])

        ####  assume the batch_size = 2  ####
        if i % 2 == 0:
            # calculate iou
            labels = labels.to(device)
            preds = preds.to(device)
            iou = MulticlassJaccardIndex(num_classes=4).to(device)
           
            iou = iou(labels, preds)
           
            correct_preds = torch.sum(preds == labels.data)
            total_preds = labels.numel()
            accuracy = correct_preds / total_preds

            ious.append(iou)
            accuracies.append(accuracy)
            print(f"IoU: {iou}; Accuracy: {accuracy}")
       
        """
        iteration 0
        IoU: 0.06368255615234375; Accuracy: 0.254730224609375
        IoU: 0.07048210501670837; Accuracy: 0.281707763671875
        IoU: 0.0514175221323967; Accuracy: 0.2054443359375
        IoU: 0.0644511952996254; Accuracy: 0.257049560546875
        IoU: 0.0889020636677742; Accuracy: 0.349578857421875
        IoU: 0.07759857177734375; Accuracy: 0.310394287109375
        IoU: 0.15658432245254517; Accuracy: 0.402587890625
        IoU: 0.08056781440973282; Accuracy: 0.315185546875
        IoU: 0.11690718680620193; Accuracy: 0.285491943359375
        IoU: 0.10627412796020508; Accuracy: 0.341949462890625
        epoch: 0 Train Loss: 5.4496 Acc: 0.2752
        """        
        ####
                
        running_corrects += torch.sum(preds == labels.data)

        loss.backward()
        optim.step()
        running_loss += loss.item() * faces.size(0)
    epoch_loss = running_loss / n_samples
    epoch_acc = running_corrects / n_samples / num_of_patch / patch_size
    print('epoch: {:} Train Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
    message = 'epoch: {:} Train Loss: {:.4f} Acc: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc)
    with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
        f.write(message)


if __name__ == '__main__':
    seed_torch(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--optim', choices=['adam', 'sgd', 'adamw'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_milestones', type=int, default=None, nargs='+',)
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_dim', type=int, default=512)
    parser.add_argument('--decoder_num_heads', type=int, default=6)
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--drop_path', type=float, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=4)
    parser.add_argument('--channels', type=int, default=10)
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--augment_deformation', action='store_true')
    parser.add_argument('--lw1', type=float, default=0.5)
    parser.add_argument('--lw2', type=float, default=0.5)
    parser.add_argument('--fpn', action='store_true')
    parser.add_argument('--face_pos', action='store_true')
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--dataset_name', type=str, default='alien', choices=['alien', 'human'])
    parser.add_argument('--seg_parts', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.25)
    args = parser.parse_args()
    mode = args.mode
    name = args.name
    dataroot = args.dataroot
    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    if args.augment_deformation:
        augments.append('deformation')
    train_dataset = SegmentationDataset(dataroot, mode='train', augments=augments)
    val_dataset = SegmentationDataset(dataroot, mode='val')
    
    print("The number of training files:", len(train_dataset))
    print("The number of valid files:", len(test_dataset))

    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True)
    val_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                       shuffle=False, pin_memory=True)
    # ========== Network ==========

    net = Mesh_baseline_seg(masking_ratio=args.mask_ratio,
                            channels=args.channels,
                            num_heads=args.heads,
                            encoder_depth=args.encoder_depth,
                            embed_dim=args.dim,
                            decoder_num_heads=args.decoder_num_heads,
                            decoder_depth=args.decoder_depth,
                            decoder_embed_dim=args.decoder_dim,
                            patch_size=args.patch_size,
                            drop_path=args.drop_path,
                            fpn=args.fpn,
                            face_pos=args.face_pos,
                            seg_part=args.seg_parts)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # ========== Optimizer ==========
    if args.optim == 'adamw':
        optim = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_milestones is not None:
        scheduler = MultiStepLR(optim, milestones=args.lr_milestones, gamma=args.gamma)
    else:

        scheduler = CosineAnnealingLR(optim, T_max=args.max_epoch, eta_min=args.lr_min, last_epoch=-1)

    criterion = nn.CrossEntropyLoss()
    checkpoint_path = os.path.join('checkpoints', name)
    #checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')

    os.makedirs(checkpoint_path, exist_ok=True)

    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint), strict=False)

    train.step = 0

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            # train_data_loader.dataset.set_epoch()
            print('iteration', epoch)
            train(net, optim, criterion, train_data_loader, epoch, args)
            print('train finished')
            #test(net, criterion, test_data_loader, epoch, args)
            #print('test finished')
            scheduler.step()
            print(optim.param_groups[0]['lr'])
