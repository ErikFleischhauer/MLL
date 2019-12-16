import datetime
import os
import random

import torch
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from torchvision.datasets import VOCSegmentation
from torch import optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


import transforms as extended_transforms
#from models import *
from misc import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d, colorize_mask
from torch.nn.modules.loss import CrossEntropyLoss
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = '../../ckpt'
exp_name = 'voc-fcn8s'

args = {
    'epoch_num': 300,
    'lr': 1e-10,
    'weight_decay': 1e-4,
    'momentum': 0.95,
    'lr_patience': 100,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes learning from scratch
    'print_freq': 20,
    'val_save_to_img_file': False,
    'val_img_sample_rate': 0.1  # randomly sample some validation results to display
}

def main(train_args, model):
    print(train_args)

    net = model.to(device)

    if len(train_args['snapshot']) == 0:
        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + train_args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, train_args['snapshot'])))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}

    net.train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    input_transform = standard_transforms.Compose([
        standard_transforms.Pad(200),
        standard_transforms.CenterCrop(320),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    train_transform = standard_transforms.Compose([
        standard_transforms.Pad(200),
        standard_transforms.CenterCrop(320),
        extended_transforms.MaskToTensor()])

    target_transform = extended_transforms.MaskToTensor()

    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage(),
    ])

    visualize = standard_transforms.Compose([
        standard_transforms.Resize(400),
        standard_transforms.CenterCrop(400),
        standard_transforms.ToTensor()
    ])

    train_set = VOCSegmentation(root='./', image_set='train', transform=input_transform, target_transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=1, num_workers=4, shuffle=True)
    val_set = VOCSegmentation(root='./', image_set='val', transform=input_transform, target_transform=train_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=False)

    criterion = CrossEntropyLoss(ignore_index=255, reduction='mean').to(device)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], momentum=train_args['momentum'])

    """optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], betas=(train_args['momentum'], 0.999))"""

    if len(train_args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + train_args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * train_args['lr']
        optimizer.param_groups[1]['lr'] = train_args['lr']

    """check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(train_args) + '\n\n')"""

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=train_args['lr_patience'], min_lr=1e-10, verbose=True)
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        train(train_loader, net, criterion, optimizer, epoch, train_args)
        val_loss, imges = validate(val_loader, net, criterion, optimizer, epoch, train_args, restore_transform, visualize)
        #imges.show()
        scheduler.step(val_loss)
    return imges

def train(train_loader, net, criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:], "inputs are " + str(inputs.size()[2:]) + "output is" + str(labels.size()[1:])
        N = inputs.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == 21

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.data, N)

        curr_iter += 1

        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg
            ))

def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        inputs = inputs.to(device)
        gts = gts.to(device)
        
        with torch.no_grad():
            outputs = net(inputs)
        
        predictions = outputs.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data / N, N)

        if random.random() > train_args['val_img_sample_rate']:
            inputs_all.append(None)
        else:
            inputs_all.append(inputs.squeeze_(0).cpu())
        gts_all.append(gts.squeeze_(0).cpu().numpy())
        predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, 21)

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']
        )
        #torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        #torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

        if train_args['val_save_to_img_file']:
            pass
            #to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
            #check_mkdir(to_save_dir)

    val_visual = []
    for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
        if data[0] is None:
            continue
        input_pil = restore(data[0])
        gt_pil = colorize_mask(data[1])
        predictions_pil = colorize_mask(data[2])
        if train_args['val_save_to_img_file']:
            pass
            #input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
            #predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
            #gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
        val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                            visualize(predictions_pil.convert('RGB'))])
    val_visual = torch.stack(val_visual, 0)
    val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)

    print('--------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch']))

    print('--------------------------------------------------------------------')

    net.train()
    return val_loss.avg, val_visual

if __name__ == '__main__':
    main(args)
