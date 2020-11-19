from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import os, shutil, json, sys
import argparse
import time

import tools.custom_loss as custom_loss
from tools.evaluation import map_and_auc, compute_distance, compute_map
from tools.misc import get_run_name, get_latest_ckpt

import tools.misc as misc
from pathlib import Path
from config import DATASETS, NUM_VIEWS
import models.ngram_sbr_net as ngvnn
import tools.provider as provider
from tensorboardX import SummaryWriter
import logging

from Dataset_Loader import get_dataloader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="pointnet")
    parser.add_argument('-dataset', type=str, default='ModelNet10', help='THe name of dataset')
    parser.add_argument('-abstract', type=float, default=0.5, help='The degree of abstractness')
    parser.add_argument("-pretrain", default=False, dest='pretrain', action='store_true')
    parser.add_argument("-num_views", type=int, help="number of views", default=12)

    parser.add_argument('-margin', type=float, default=2.0, help='margin for triplet center loss')
    parser.add_argument('-w1', type=float, default=1, help='weight for classification loss')
    parser.add_argument('-w2', type=float, default=0.0)
    parser.add_argument('-n_classes', type=int, default=8)
    parser.add_argument('-n_samples', type=int, default=1)

    parser.add_argument(
        '-resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument("-weight_decay", type=float, help="weight decay", default=1e-4)
    parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage",
                        default=4)  # it will be *12 images in each batch for mvcnn
    parser.add_argument("-epoch", type=int, help="epoch", default=1)
    parser.add_argument('-gradient_clip', type=float, default=0.05)  # previous i set it to be 0.01
    parser.add_argument('-print-freq', type=int, default=50)

    parser.add_argument("-submit", action='store_true', help="whether submitted", default=False)
    parser.add_argument("-triplet_type", type=str, help="Name of the experiment", default="tpl")

    parser.add_argument('-model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('-num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('-uniform', action='store_true', default=False, help='Whether to use uniform [default: False]')
    parser.add_argument("-view_type", type=str, default='depth')
    parser.add_argument('-batch_size', type=int, default=8, help='batch size in training [default: 24]')
    parser.add_argument("-network_target", action='store_true', help="whether submitted", default=False)
    parser.add_argument("-sketch_target", type=str, default='')
    parser.add_argument("-sketch_dir", type=str, default='coverage20_modelnet_merge_update')
    return parser.parse_args()

def log_string(str, logger):
    logger.info(str)
    print(str)

def main(args):
    if args.name == 'sbr':
        experiment_dir = Path(os.path.join(BASE_DIR, 'save/sbr/' + get_run_name()))
    else:
        experiment_dir = Path(os.path.join(BASE_DIR, 'save/double_{}/'.format(args.name) + get_run_name()))

    experiment_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    # writer = SummaryWriter(log_dir)


    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    num_class = DATASETS[args.dataset]['n_classes']

    if args.name == 'sbr':
        net_p = ngvnn.Net_Prev(pretraining=args.pretrain, num_views=args.num_views)
        net_whole = ngvnn.Net_Whole(pretraining=args.pretrain)
    elif args.name == 'pointnet':
        import importlib
        model = importlib.import_module(args.model)
        net_p = model.get_encoder_model(num_class, normal_channel=False)
        net_whole = model.get_encoder_model(num_class, normal_channel=False)
    elif args.name == 'ngvnn':
        net_p = ngvnn.Net_Prev(pretraining=args.pretrain, num_views=args.num_views)
        net_whole = ngvnn.Net_Prev(pretraining=args.pretrain, num_views=args.num_views)
    else:
        NotImplementedError

    net_cls = ngvnn.Net_Classifier(nclasses=num_class)

    crt_cls = nn.CrossEntropyLoss().cuda()
    if args.triplet_type == 'tcl':
        center_embed = 512
        crt_tpl = custom_loss.TripletCenterLoss(margin=args.margin, center_embed=center_embed, num_classes=num_class).cuda()
        optim_centers = torch.optim.SGD(crt_tpl.parameters(), lr=0.1)
    else:
        from dataset.TripletSampler import HardestNegativeTripletSelector
        anchor_index = args.n_classes * args.n_samples
        crt_tpl = custom_loss.OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin, False, anchor_index))
    criterion = [crt_cls, crt_tpl, args.w1, args.w2]
    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        net_p.load_state_dict(checkpoint['net_p'])
        net_whole.load_state_dict(checkpoint['net_whole'])
        net_cls.load_state_dict(checkpoint['net_cls'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_prec']


    net_whole = nn.DataParallel(net_whole).cuda()
    net_cls = nn.DataParallel(net_cls).cuda()
    net_p = nn.DataParallel(net_p).cuda()

    optim_shape = optim.SGD([{'params': net_p.parameters()},
                             {'params': net_cls.parameters()}],
                            lr=0.001, momentum=0.9, weight_decay=args.weight_decay)
    if args.name in ['sbr', 'ngvnn', 'pointnet']:
        base_param_ids = set(map(id, net_whole.module.features.parameters()))
        new_params = [p for p in net_whole.parameters() if id(p) not in base_param_ids]
        param_groups = [
            {'params': net_whole.module.features.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

        optim_sketch = optim.SGD(param_groups, lr=0.001, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optim_sketch = optim.SGD([{'params': net_p.parameters()},
                                 {'params': net_cls.parameters()}],
                                lr=0.001, momentum=0.9, weight_decay=args.weight_decay)

    if args.triplet_type == 'tcl':
        optimizer = (optim_sketch, optim_shape, optim_centers)
    else:
        optimizer = (optim_sketch, optim_shape)
    model = (net_whole, net_p, net_cls)

    # Schedule learning rate
    def adjust_lr(epoch, optimizer):
        step_size = 800 if args.pk_flag else 80  # 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    train_shape_loader, train_sketch_loader, test_shape_loader, test_sketch_loader = get_dataloader(args)
    # Start training
    top1 = 0.0
    best_epoch = -1
    best_metric = None

    '''TRANING'''
    logger.info('Start training...')

    for epoch in range(start_epoch, args.epoch):
        # cls acc top1
        train_top1 = train(train_sketch_loader, train_shape_loader, model, criterion, optimizer, epoch, args, logger)
        if train_top1 > 0.1:
            print("Test:")
            cur_metric = validate(test_sketch_loader, test_shape_loader, model, criterion, args, logger)
            top1 = cur_metric[3] # mAP_feat_norm

        is_best = top1 > best_top1
        if is_best:
            best_epoch = epoch + 1
            best_metric = cur_metric
        best_top1 = max(top1, best_top1)

        # path_checkpoint = '{0}/model_latest.pth'.format(checkpoints_dir)
        # misc.save_checkpoint(checkpoint, path_checkpoint)

        if is_best:  # save checkpoint
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            checkpoint = {}
            checkpoint['epoch'] = epoch + 1
            checkpoint['current_prec'] = top1
            checkpoint['best_prec'] = best_top1
            checkpoint['net_p'] = net_p.module.state_dict()
            checkpoint['net_whole'] = net_whole.module.state_dict()
            checkpoint['net_cls'] = net_cls.module.state_dict()

            # torch.save(checkpoint, savepath)
            # path_checkpoint = '{0}/best_model.pth'.format(checkpoints_dir)
            misc.save_checkpoint(checkpoint, savepath)#path_checkpoint)


        log_string('\n * Finished epoch {:3d}  top1: {:5.3%}  best: {:5.3%}{} @epoch {}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else '', best_epoch), logger)

    logger.info('End of training...')

    log_string('Best metric {}'.format(best_metric), logger)

    return experiment_dir



def train(sketch_dataloader, shape_dataloader, model, criterion, optimizer, epoch, args, logger):
    """
    train for one epoch on the training set
    """
    batch_time = misc.AverageMeter()
    losses = misc.AverageMeter()
    top1 = misc.AverageMeter()
    tpl_losses = misc.AverageMeter()

    # training mode
    net_whole, net_p, net_cls = model
    if args.triplet_type == 'tcl':
        optim_sketch, optim_shape, optim_centers = optimizer
    else:
        optim_sketch, optim_shape = optimizer
    crt_cls, crt_tpl, w1, w2 = criterion

    net_whole.train()
    net_cls.train()

    end = time.time()


    for i, ((sketches, k_labels), (shapes, p_labels)) in enumerate(zip(sketch_dataloader, shape_dataloader)):
        if args.name == 'ngvnn':
            sketches = sketches.view(sketches.size(0) * sketches.size(1), sketches.size(2), sketches.size(3), sketches.size(4))
            # expanding: (bz * 12) x 3 x 224 x 224
            sketches = sketches.expand(sketches.size(0), 3, sketches.size(2), sketches.size(3))
            shapes = shapes.view(shapes.size(0) * shapes.size(1), shapes.size(2), shapes.size(3), shapes.size(4))
            # expanding: (bz * 12) x 3 x 224 x 224
            shapes = shapes.expand(shapes.size(0), 3, shapes.size(2), shapes.size(3))
        elif args.name == 'pointnet':
            sketches = provider.random_point_dropout(sketches.data.numpy())
            sketches[:, :, 0:3] = provider.random_scale_point_cloud(sketches[:, :, 0:3])
            sketches[:, :, 0:3] = provider.shift_point_cloud(sketches[:, :, 0:3])
            sketches = torch.Tensor(sketches).transpose(2, 1)

            shapes = provider.random_point_dropout(shapes.data.numpy())
            shapes[:, :, 0:3] = provider.random_scale_point_cloud(shapes[:, :, 0:3])
            shapes[:, :, 0:3] = provider.shift_point_cloud(shapes[:, :, 0:3])
            shapes = torch.Tensor(shapes).transpose(2, 1)
        elif args.name == 'sbr':
            shapes = shapes.view(shapes.size(0) * shapes.size(1), shapes.size(2), shapes.size(3), shapes.size(4))
            # expanding: (bz * 12) x 3 x 224 x 224
            shapes = shapes.expand(shapes.size(0), 3, shapes.size(2), shapes.size(3))
        else:
            NotImplementedError

        shapes_v = Variable(shapes.cuda())
        p_labels_v = Variable(p_labels.long().cuda())

        sketches_v = Variable(sketches.cuda())
        k_labels_v = Variable(k_labels.long().cuda())

        shape_feat = net_p(shapes_v)
        sketch_feat = net_whole(sketches_v)
        feat = torch.cat([sketch_feat, shape_feat])
        target = torch.cat([k_labels_v, p_labels_v])
        score = net_cls(feat)

        cls_loss = crt_cls(score, target)

        tpl_loss, triplet_num = crt_tpl(feat, target)

        if args.triplet_type == 'tcl':
            optim_centers.zero_grad()

        loss = w1 * cls_loss + w2 * tpl_loss
        ## measure accuracy
        prec1 = misc.accuracy(score.data, target.data, topk=(1,))[0]
        losses.update(cls_loss.data, score.shape[0])  # batchsize

        tpl_losses.update(tpl_loss.data, score.size(0))
        top1.update(prec1, score.shape[0])

        ## backward
        optim_sketch.zero_grad()
        optim_shape.zero_grad()
        if args.triplet_type == 'tcl':
            optim_centers.zero_grad()

        loss.backward()
        misc.clip_gradient(optim_sketch, args.gradient_clip)
        misc.clip_gradient(optim_shape, args.gradient_clip)
        if args.triplet_type == 'tcl':
            misc.clip_gradient(optim_centers, args.gradient_clip)

        optim_sketch.step()
        optim_shape.step()
        if args.triplet_type == 'tcl':
            optim_centers.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            log_string('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Trploss {triplet.val:.4f}({triplet.avg:.3f})\t'
                  'Triplet num {3}'.format(
                epoch, i, len(sketch_dataloader), triplet_num, batch_time=batch_time,
                loss=losses, top1=top1, triplet=tpl_losses), logger)
    log_string(' * Train Prec@1 {top1.avg:.3f}'.format(top1=top1), logger)
    return top1.avg


def validate(sketch_dataloader, shape_dataloader, model, criterion, args, logger):
    """
    test for one epoch on the testing set
    """
    sketch_losses = misc.AverageMeter()
    sketch_top1 = misc.AverageMeter()

    shape_losses = misc.AverageMeter()
    shape_top1 = misc.AverageMeter()

    net_whole, net_p, net_cls = model
    crt_cls, crt_tl, w1, w2 = criterion

    net_whole.eval()
    net_p.eval()
    net_cls.eval()

    sketch_features = []
    sketch_scores = []
    sketch_labels = []

    shape_features = []
    shape_scores = []
    shape_labels = []

    batch_time = misc.AverageMeter()
    end = time.time()

    for i, (sketches, k_labels) in enumerate(sketch_dataloader):
        if args.name == 'ngvnn':
            sketches = sketches.view(sketches.size(0) * sketches.size(1), sketches.size(2), sketches.size(3), sketches.size(4))
            # expanding: (bz * 12) x 3 x 224 x 224
            sketches = sketches.expand(sketches.size(0), 3, sketches.size(2), sketches.size(3))
        elif args.name == 'pointnet':
            sketches = provider.random_point_dropout(sketches.data.numpy())
            sketches[:, :, 0:3] = provider.random_scale_point_cloud(sketches[:, :, 0:3])
            sketches[:, :, 0:3] = provider.shift_point_cloud(sketches[:, :, 0:3])
            sketches = torch.Tensor(sketches).transpose(2, 1)

        sketches_v = Variable(sketches.cuda())
        k_labels_v = Variable(k_labels.long().cuda())
        sketch_feat = net_whole(sketches_v)
        sketch_score = net_cls(sketch_feat)

        loss = crt_cls(sketch_score, k_labels_v)

        prec1 = misc.accuracy(sketch_score.data, k_labels_v.data, topk=(1,))[0]
        sketch_losses.update(loss.data, sketch_score.shape[0])  # batchsize

        sketch_top1.update(prec1, sketch_score.shape[0])

        sketch_features.append(sketch_feat.data.cpu())
        sketch_labels.append(k_labels)
        sketch_scores.append(sketch_score.data.cpu())

        batch_time.update(time.time() - end)

        end = time.time()
        if i % args.print_freq == 0:
            log_string('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(sketch_dataloader), batch_time=batch_time, loss=sketch_losses,
                top1=sketch_top1), logger)
    log_string(' *Sketch Prec@1 {top1.avg:.3f}'.format(top1=sketch_top1), logger)

    batch_time = misc.AverageMeter()
    end = time.time()
    for i, (shapes, p_labels) in enumerate(shape_dataloader):
        if args.name == 'pointnet':
            shapes = shapes.transpose(2, 1)
        else:
            shapes = shapes.view(shapes.size(0) * shapes.size(1), shapes.size(2), shapes.size(3), shapes.size(4))
            # expanding: (bz * 12) x 3 x 224 x 224
            shapes = shapes.expand(shapes.size(0), 3, shapes.size(2), shapes.size(3))

        shapes_v = Variable(shapes.cuda())
        p_labels_v = Variable(p_labels.long().cuda())

        shape_feat = net_p(shapes_v)
        shape_score = net_cls(shape_feat)

        loss = crt_cls(shape_score, p_labels_v)

        prec1 = misc.accuracy(shape_score.data, p_labels_v.data, topk=(1,))[0]
        shape_losses.update(loss.data, shape_score.shape[0])  # batchsize
        shape_top1.update(prec1, shape_score.shape[0])

        shape_features.append(shape_feat.data.cpu())
        shape_labels.append(p_labels)
        shape_scores.append(shape_score.data.cpu())

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_string('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(shape_dataloader), batch_time=batch_time, loss=shape_losses,
                top1=shape_top1), logger)
    log_string(' *Shape Prec@1 {top1.avg:.3f}'.format(top1=shape_top1), logger)

    shape_features = torch.cat(shape_features, 0).numpy()
    sketch_features = torch.cat(sketch_features, 0).numpy()

    shape_scores = torch.cat(shape_scores, 0).numpy()
    sketch_scores = torch.cat(sketch_scores, 0).numpy()

    shape_labels = torch.cat(shape_labels, 0).numpy()
    sketch_labels = torch.cat(sketch_labels, 0).numpy()

    d_feat = compute_distance(sketch_features.copy(), shape_features.copy(), l2=False)
    d_feat_norm = compute_distance(sketch_features.copy(), shape_features.copy(), l2=True)
    mAP_feat = compute_map(sketch_labels.copy(), shape_labels.copy(), d_feat)
    mAP_feat_norm = compute_map(sketch_labels.copy(), shape_labels.copy(), d_feat_norm)
    log_string(' * Feature mAP {0:.5%}\tNorm Feature mAP {1:.5%}'.format(mAP_feat, mAP_feat_norm), logger)

    d_score = compute_distance(sketch_scores.copy(), shape_scores.copy(), l2=False)
    mAP_score = compute_map(sketch_labels.copy(), shape_labels.copy(), d_score)
    d_score_norm = compute_distance(sketch_scores.copy(), shape_scores.copy(), l2=True)
    mAP_score_norm = compute_map(sketch_labels.copy(), shape_labels.copy(), d_score_norm)
    log_string(' * Score mAP {0:.5%}\tNorm Score mAP {1:.5%}'.format(mAP_score, mAP_score_norm), logger)
    return [sketch_top1.avg, shape_top1.avg, mAP_feat, mAP_feat_norm, mAP_score, mAP_score_norm]


if __name__ == '__main__':
    args = parse_args()
    experiment_dir = main(args)
    os.system('sh run_eval_all.sh 5 %s' % experiment_dir)
