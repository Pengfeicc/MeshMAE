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


def test(net, criterion, test_dataset, epoch, args):

    net.eval()
    acc = 0
    running_loss = 0
    running_corrects = 0
    n_samples = 0
    test_accuracies, test_ious = [], []
    for i, (face_patch, feats_patch, np_Fs, center_patch, coordinate_patch, labels, filename) in enumerate(
            test_dataset):
        filename = filename[0]
        filename = filename.split("/")[-1][:-4]

        faces = face_patch.cuda()
        feats = feats_patch.to(torch.float32).cuda()
        centers = center_patch.to(torch.float32).cuda()
        Fs = np_Fs.cuda()
        cordinates = coordinate_patch.to(torch.float32).cuda()

        labels = labels.to(torch.long).cuda()
        labels = labels.reshape(faces.shape[0], -1)
        n_samples += faces.shape[0]
        with torch.no_grad():
            outputs, outputs_seg = net(faces, feats, centers, Fs, cordinates)
        outputs = outputs.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)
        outputs_seg = outputs_seg.reshape(faces.shape[0], -1, args.seg_parts).permute(0, 2, 1)

        loss = criterion(outputs, labels)
        loss_seg = criterion(outputs_seg, labels)
        loss = 0.5 * loss + 0.5 * loss_seg
        _, preds = torch.max(outputs_seg, 1)

         # calculate iou & accuracy per batch size 
        if i % 1 == 0:
            test_labels = labels.to(device)
            test_preds = preds.to(device)
            # average valid_IoU score for classes
            test_iou = MulticlassJaccardIndex(num_classes=4).to(device)
           
            correct_preds = torch.sum(test_preds == test_labels.data)
            total_preds = test_labels.numel() 
            test_accuracy = correct_preds / total_preds

            test_ious.append(test_iou(test_labels, test_preds))
            test_accuracies.append(test_accuracy)
            print(f"Valid_IoU: {test_iou(test_labels, test_preds)}; Valid_Accuracy: {test_accuracy}")

        preds_cc = preds.cpu().detach().numpy().flatten()
        np.savetxt(f"predictions/{filename}.txt", preds_cc, newline="\n", fmt="%d")

        running_corrects += torch.sum(preds == labels.data)
        running_loss += loss.item() * faces.size(0)

    epoch_acc = running_corrects.double() / n_samples / 16384
    epoch_loss = running_loss / n_samples
    return epoch_acc, epoch_loss


if __name__ == '__main__':
    seed_torch(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test', 'val'])
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
    #train_dataset = SegmentationDataset(dataroot, mode='train', augments=augments)
    val_dataset = SegmentationDataset(dataroot, mode='test')  #path: datasets/alien_small/test

    print("The number of files in test dataset:", len(val_dataset))

    test_data_loader = data.DataLoader(val_dataset, num_workers=args.n_worker, batch_size=args.batch_size,
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

    os.makedirs(checkpoint_path, exist_ok=True)

    if args.checkpoint is not None:
        net.load_state_dict(torch.load(args.checkpoint), strict=False)

    
    test.best_acc = 0

    if args.mode == 'test':
        for epoch in range(args.n_epoch):
            test_epoch_acc, test_epoch_loss = test(net, criterion, test_data_loader, 0, args)
            print('epoch: {:} test Loss: {:.4f} Acc: {:.4f}'.format(epoch, test_epoch_loss, test_epoch_acc))
            message = 'epoch: {:} test Loss: {:.4f} Acc: {:.4f}\n'.format(epoch, test_epoch_loss, test_epoch_acc)
    
