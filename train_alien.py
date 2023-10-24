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
import sys
from metrics import jaccard_index
from torchmetrics.classification import MulticlassJaccardIndex

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
    training_loss = 0
    training_corrects = 0
    train_n_samples = 0
    patch_size = 64
    num_of_patch = 0
    train_accuracies, train_ious = [], []
    # training loop
    for i, (face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, train_labels) in enumerate(train_dataset):
        optim.zero_grad() # 梯度清零
        train_faces = face_patch.cuda() 
        patch_size = train_faces.size(2)
        num_of_patch = train_faces.size(1) 

        train_feats = feats_patch.to(torch.float32).cuda()
        train_centers = center_patch.to(torch.float32).cuda() 
        train_Fs = np_Fs.cuda() # torch.Size([2, 256, 64, 1])
        train_cordinates = coordinate_patch.to(torch.float32).cuda()
        train_labels = train_labels.to(torch.long).cuda()

        train_labels = train_labels.reshape(train_faces.shape[0], -1) 

        train_n_samples += train_faces.shape[0] 

        train_outputs, train_outputs_seg = net(train_faces, train_feats, train_centers, train_Fs, train_cordinates) 
        train_outputs = train_outputs.reshape(train_faces.shape[0], -1, args.seg_parts).permute(0, 2, 1) # torch.Size([2, 4, 16384])
        train_outputs_seg = train_outputs_seg.reshape(train_faces.shape[0], -1, args.seg_parts).permute(0, 2, 1) # torch.Size([2, 4, 16384])

        dim = train_outputs.shape[1]

        train_loss = criterion(train_outputs, train_labels) 
        train_loss_seg = criterion(train_outputs_seg, train_labels) 
        train_total_loss = args.lw1 * train_loss + args.lw2 * train_loss_seg

        _, train_preds = torch.max(train_outputs_seg, 1) # torch.Size([2, 16384])
        
        # calculate iou & accuracy per batch size
        if i % 2 == 0:         
            train_labels = train_labels.to(device)
            train_preds = train_preds.to(device)
            # average train_IoU score for classes
            train_iou = MulticlassJaccardIndex(num_classes=4).to(device)
           
            correct_preds = torch.sum(train_preds == train_labels.data)
            total_preds = train_labels.numel() #计算labels中元素的总数
            train_accuracy = correct_preds / total_preds

            train_ious.append(train_iou(train_labels, train_preds))
            train_accuracies.append(train_accuracy)
            print(f"Train_IoU: {train_iou(train_labels, train_preds)}; Train_Accuracy: {train_accuracy}")
        training_corrects += torch.sum(train_preds == train_labels.data)

        train_total_loss.backward()
        optim.step()
        training_loss += train_total_loss.item() * train_faces.size(0)

    epoch_loss = training_loss / train_n_samples
    epoch_acc = training_corrects / train_n_samples / num_of_patch / patch_size
    return epoch_loss, epoch_acc

    
def valid(net, criterion, test_dataset, epoch, args):
    net.eval()
    acc = 0
    validing_loss = 0
    validing_corrects = 0
    patch_size = 64
    valid_n_samples = 0
    valid_accuracies, valid_ious = [], []
    # validation loop
    with torch.no_grad():
        for i, (face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, valid_labels, filename) in enumerate(test_dataset):
            #face_patch_tensor = torch.from_numpy(face_patch)
            valid_faces = face_patch.cuda()
            patch_size = valid_faces.size(2) 
            num_of_patch = valid_faces.size(1) 

            valid_feats = feats_patch.to(torch.float32).cuda()
            valid_centers = center_patch.to(torch.float32).cuda()
            valid_Fs = np_Fs.cuda() 
            valid_cordinates = coordinate_patch.to(torch.float32).cuda()

            valid_labels = valid_labels.to(torch.long).cuda()
            valid_labels = valid_labels.reshape(valid_faces.shape[0], -1)

            valid_n_samples += valid_faces.shape[0]

            valid_outputs, valid_outputs_seg = net(valid_faces, valid_feats, valid_centers, valid_Fs, valid_cordinates)
            valid_outputs = valid_outputs.reshape(valid_faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)
            valid_outputs_seg = valid_outputs_seg.reshape(valid_faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)

            valid_loss = criterion(valid_outputs, valid_labels)
            valid_loss_seg = criterion(valid_outputs_seg, valid_labels)
            valid_total_loss = args.lw1 * valid_loss + args.lw2 * valid_loss_seg

            _, valid_preds = torch.max(valid_outputs_seg, 1)

            # calculate iou & accuracy per batch size 
            if i % 1 == 0:
                valid_labels = valid_labels.to(device)
                valid_preds = valid_preds.to(device)
                # average valid_IoU score for classes
                valid_iou = MulticlassJaccardIndex(num_classes=4).to(device)
           
                correct_preds = torch.sum(valid_preds == valid_labels.data)
                total_preds = valid_labels.numel() #计算labels中元素的总数
                valid_accuracy = correct_preds / total_preds

                valid_ious.append(valid_iou(valid_labels, valid_preds))
                valid_accuracies.append(valid_accuracy)
                print(f"Valid_IoU: {valid_iou(valid_labels, valid_preds)}; Valid_Accuracy: {valid_accuracy}")
        validing_corrects += torch.sum(valid_preds == valid_labels.data)
        validing_loss += valid_total_loss.item() * valid_faces.size(0)

    epoch_loss = validing_loss / valid_n_samples
    epoch_acc = validing_corrects / valid_n_samples / num_of_patch / patch_size
    return epoch_loss, epoch_acc


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
    test_dataset = SegmentationDataset(dataroot, mode='val')   # path: datasets/alien_small/val
    print("The number of files in train dataset:", len(train_dataset))  # 40 files
    print("The number of files in valid dataset:", len(test_dataset))   # 5  flies

    train_data_loader = data.DataLoader(train_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
                                        shuffle=True, pin_memory=True)
    valid_data_loader = data.DataLoader(test_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
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
    logs = checkpoint_path
    writer = SummaryWriter(logs)


    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            # train_data_loader.dataset.set_epoch()
            print('iteration', epoch)
            train_epoch_loss, train_epoch_acc = train(net, optim, criterion, train_data_loader, epoch, args)
            valid_epoch_loss, valid_epoch_acc = valid(net, criterion, valid_data_loader, epoch, args)

            print('/n epoch: {:} Train Loss: {:.4f} Train Acc: {:.4f} Valid Acc: {:.4f}'.format(epoch, train_epoch_loss, train_epoch_acc, valid_epoch_acc))
            message = 'epoch: {:} Train Loss: {:.4f} Train Acc: {:.4f} Valid Acc: {:.4f}\n'.format(epoch, train_epoch_loss, train_epoch_acc, valid_epoch_acc)
            with open(os.path.join('checkpoints', name, 'log.txt'), 'a') as f:
                f.write(message)

            scheduler.step()
            print('Learning Rate:', optim.param_groups[0]['lr'])
            print('train finished')

            writer.add_scalar('Loss', train_epoch_loss, global_step=epoch, walltime=epoch)
            writer.add_scalar('Loss', valid_epoch_loss, global_step=epoch, walltime=epoch)
            writer.add_scalar('Accuracy', train_epoch_acc, global_step=epoch, walltime=epoch)
            writer.add_scalar('Accuracy', valid_epoch_acc, global_step=epoch, walltime=epoch)
            
        # save the weights
        model_weight = copy.deepcopy(net.state_dict())
        torch.save(model_weight, os.path.join('checkpoints', name, 'best.pkl'))
        
