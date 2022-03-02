import argparse
import numpy as np
import os, json
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import tools.provider as provider
import socket
import tools.misc as misc
from tensorboardX import SummaryWriter
import time
from tools.evaluation import map_and_auc, compute_distance, compute_map
from torch.autograd import Variable
import chamfer_python
from Dataset_Loader import get_dataloader
from args import get_args


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat, 0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat, 1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:, 2] =  class_acc[:, 0]/ class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc

def log_string(str, logger):
    logger.info(str)
    print(str)

def main(args):

    # '''HYPER PARAMETER'''
    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    # experiment_dir = Path('./log/')
    experiment_dir = Path(os.path.join(BASE_DIR, 'save/'))

    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.name)
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr + '_' + socket.gethostname())
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir)


    '''LOG'''
    args.abstract = [float(item) for item in args.abstract.split(',')]
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
    # log_string('PARAMETER ...')
    # log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...', logger)
    train_shape_loader, train_sketch_loader, test_shape_loader, test_sketch_loader = get_dataloader(args)

    '''MODEL LOADING'''
    num_class = args.num_class

    if args.name == 'pointnet':
        import importlib
        model = importlib.import_module(args.model)
        classifier = model.get_model(num_class, num_points=args.num_point, decoder=args.reconstruct).cuda()
    elif args.name == 'ngvnn':
        from models.n_gram import NGVNN
        classifier = NGVNN(args.name, nclasses=num_class, pretraining=args.pretrain, cnn_name=args.cnn_name).cuda()
    else:
        NotImplementedError


    crt_cls = torch.nn.CrossEntropyLoss().cuda()
    import tools.custom_loss as custom_loss
    if args.triplet_type == 'tcl':
        center_embed = 512
        crt_tpl = custom_loss.TripletCenterLoss(margin=args.margin, center_embed=center_embed, num_classes=num_class).cuda()
        optim_centers = torch.optim.SGD(crt_tpl.parameters(), lr=0.1)
    else:
        from dataset.TripletSampler import HardestNegativeTripletSelector
        anchor_index = args.n_classes * args.n_samples
        crt_tpl = custom_loss.OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin, args.sketch_anchor, anchor_index))

    if args.reconstruct:
        # if torch.cuda.is_available():
        #     from chamfer_distance.chamfer_distance_gpu import ChamferDistance  # https://github.com/chrdiller/pyTorchChamferDistance
        # else:
        #     from chamfer_distance.chamfer_distance_cpu import ChamferDistance  # https://github.com/chrdiller/pyTorchChamferDistance
        # import chamfer3D.dist_chamfer_3D
        import chamfer_python
        # chamfer_dist = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        criterion = [crt_cls, crt_tpl]#, chamfer_dist]
    else:
        criterion = [crt_cls, crt_tpl]

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model', logger)
    except:
        log_string('No existing model, starting training from scratch...', logger)
        start_epoch = 0
        start_step = 0

    if args.optimizer == 'Adam':
        optim = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optim = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    if args.triplet_type == 'tcl':
        optimizer = (optim, optim_centers)
    else:
        optimizer = optim

    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.7)
    global_epoch = 0
    top1 = 0.0
    best_epoch = -1
    best_metric = None

    '''TRANING'''
    logger.info('Start training...')
    best_top1 = 0
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch), logger)

        train_top1, end_step = train(logger, train_sketch_loader, train_shape_loader, classifier, criterion, optimizer, writer, start_step, epoch, args)
        start_step = end_step
        log_string('train_top1: %f' % train_top1, logger)
        scheduler.step()

        # plot learning rate
        lr = optim.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('params/lr', lr, epoch)

        if train_top1 > 0.1:
            log_string("Test:", logger)
            cur_metric = validate(logger, test_sketch_loader, test_shape_loader, classifier, criterion)
            top1 = cur_metric[3] # mAP_feat_norm

        is_best = top1 > best_top1
        if is_best:
            best_epoch = epoch + 1
            best_metric = cur_metric
        best_top1 = max(top1, best_top1)

        writer.add_scalar('val/val_feature_map', cur_metric[3], epoch + 1) # mAP_feat_norm
        writer.add_scalar('val/val_score_map', cur_metric[-1], epoch + 1)

        if is_best:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath, logger)
            state = {
                'epoch': best_epoch,
                'step': start_step,
                'current_prec': top1,
                'best_prec': best_top1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }
            torch.save(state, savepath)

        log_string('\n * Finished epoch {:3d}  top1: {:5.3%}  best: {:5.3%}{} @epoch {}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else '', best_epoch), logger)

        global_epoch += 1


    logger.info('End of training...')
    writer.export_scalars_to_json(log_dir.joinpath("all_scalars.json"))
    writer.close()

    log_string('Best metric {}'.format(best_metric), logger)

    return experiment_dir

def train(logger, sketch_dataloader, shape_dataloader, model, criterion, optimizer, writer, start_step, epoch, args):
    model = model.train()

    batch_time = misc.AverageMeter()
    losses = misc.AverageMeter()
    top1 = misc.AverageMeter()
    tpl_losses = misc.AverageMeter()
    rec_losses = misc.AverageMeter()

    if args.reconstruct:
        crt_cls, crt_tpl = criterion#, chamfer_dist = criterion
        recon_index = args.n_classes * args.n_samples
    else:
        crt_cls, crt_tpl = criterion

    end = time.time()

    if args.triplet_type == 'tcl':
        optim, optim_centers = optimizer
    else:
        optim = optimizer

    for i, (sketch_list, (shapes, p_labels)) in enumerate(zip(sketch_dataloader, shape_dataloader)):
        if args.sketch_target == '':
            sketches, k_labels = sketch_list
        else:
            sketches, targets, k_labels = sketch_list
        if args.name == 'pointnet':
            if args.sketch_target == '':
                points = torch.cat([sketches, shapes]).data.numpy()
                target = torch.cat([k_labels, p_labels])
            else:
                points = torch.cat([sketches, targets, shapes]).data.numpy()
                target = torch.cat([k_labels, k_labels, p_labels])

            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            if args.reconstruct:
                if args.sketch_target == '': # reconstruct shape
                    pred, feat, recon_points = model(points, recon_index=[recon_index, 2 * recon_index]) # reconstruct shape
                    targets = points[recon_index:,:,:].transpose(1, 2)
                else: # reconstruct sketch
                    pred, feat, recon_points = model(points, recon_index=[0, recon_index]) # reconstruct sketch
                    targets = targets.cuda()
                recon_points = recon_points.transpose(1, 2)
                # dist1, dist2, _, _ = chamfer_dist(points, recon_points)  # calculate loss
                dist1, dist2, _, _ = chamfer_python.distChamfer(targets, recon_points)
                rec_loss = (torch.mean(dist1)) + (torch.mean(dist2))
                writer.add_scalar('train/recon_loss', rec_loss, start_step + i + 1)
                rec_losses.update(rec_loss.data, recon_index)
            else:
                pred, feat = model(points)
        elif args.name == 'ngvnn':
            views = torch.cat([sketches, shapes])
            target = torch.cat([k_labels, p_labels])
            N, V, C, H, W = views.size()
            in_data = Variable(views).view(-1, C, H, W).cuda()
            target = Variable(target).cuda()
            pred, feat = model(in_data)
        else:
            NotImplementedError

        cls_loss = crt_cls(pred, target.long())
        writer.add_scalar('train/cls_loss', cls_loss, start_step + i + 1)

        tpl_loss, triplet_num = crt_tpl(feat, target)

        tpl_losses.update(tpl_loss.data, pred.size(0))
        writer.add_scalar('train/tpl_loss', tpl_losses.val, start_step + i + 1)


        if args.reconstruct:
            loss = args.w1 * cls_loss + args.w2 * tpl_loss + args.w3 * rec_loss
        else:
            loss = args.w1 * cls_loss + args.w2 * tpl_loss
        losses.update(cls_loss.data, pred.shape[0])  # batchsize
        writer.add_scalar('train/train_loss', losses.val, start_step + i + 1)

        prec1 = misc.accuracy(pred.data, target.data, topk=(1,))[0]
        top1.update(prec1, pred.shape[0])
        # print(np.mean(correct.item() / float(points.size()[0])))
        writer.add_scalar('train/train_overall_acc', top1.val,
                          start_step + i + 1)

        ## backward
        optim.zero_grad()
        if args.triplet_type == 'tcl':
            optim_centers.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), args.gradient_clip)
        if args.triplet_type == 'tcl':
            misc.clip_gradient(optim_centers, args.gradient_clip)

        optim.step()
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
                #   'Triplet num {triplet_num:.2f}\t'
                  'Reconloss {recon.val: .4f}({recon.avg:.3f})'.format(
                epoch, i, len(sketch_dataloader), triplet_num=triplet_num, batch_time=batch_time,
                loss=losses, top1=top1, triplet=tpl_losses, recon=rec_losses), logger)

    end_step = start_step + i

    return top1.avg, end_step

def validate(logger, sketch_dataloader, shape_dataloader, model, criterion):
    sketch_losses = misc.AverageMeter()
    sketch_top1 = misc.AverageMeter()

    shape_losses = misc.AverageMeter()
    shape_top1 = misc.AverageMeter()

    crt_cls, crt_tl = criterion

    sketch_features = []
    sketch_scores = []
    sketch_labels = []

    shape_features = []
    shape_scores = []
    shape_labels = []

    batch_time = misc.AverageMeter()
    end = time.time()

    model = model.eval()
    with torch.no_grad():
        for i, (sketches, k_labels) in enumerate(sketch_dataloader):
            if args.name == 'pointnet':
                points = sketches.transpose(2, 1)
                points, k_labels_v = points.cuda(), k_labels.cuda()
                if args.reconstruct:
                    sketch_score, sketch_feat, _ = model(points)
                else:
                    sketch_score, sketch_feat = model(points)
            elif args.name == 'ngvnn':
                N, V, C, H, W = sketches.size()
                in_data = Variable(sketches).view(-1, C, H, W).cuda()
                k_labels_v = Variable(k_labels).cuda()
                sketch_score, sketch_feat = model(in_data)
            else:
                NotImplementedError

            loss = crt_cls(sketch_score, k_labels_v)

            prec1 = misc.accuracy(sketch_score.data, k_labels_v.data, topk=(1,))[0]
            sketch_losses.update(loss.data, sketch_score.shape[0])  # batchsize
            sketch_top1.update(prec1, sketch_score.shape[0])

            sketch_features.append(sketch_feat.data.cpu())
            sketch_labels.append(k_labels)
            sketch_scores.append(sketch_score.data.cpu())

            batch_time.update(time.time() - end)

        log_string(' *Sketch Prec@1 {top1.avg:.3f}'.format(top1=sketch_top1), logger)

        for i, (shapes, p_labels) in enumerate(shape_dataloader):
            if args.name == 'pointnet':
                points = shapes.transpose(2, 1)
                points, p_labels_v = points.cuda(), p_labels.cuda()
                if args.reconstruct:
                    shape_score, shape_feat, _ = model(points)
                else:
                    shape_score, shape_feat = model(points)
            elif args.name == 'ngvnn':
                N, V, C, H, W = shapes.size()
                in_data = Variable(shapes).view(-1, C, H, W).cuda()
                p_labels_v = Variable(p_labels).cuda()
                shape_score, shape_feat = model(in_data)
            else:
                NotImplementedError

            loss = crt_cls(shape_score, p_labels_v)

            prec1 = misc.accuracy(shape_score.data, p_labels_v.data, topk=(1,))[0]
            shape_losses.update(loss.data, shape_score.shape[0])  # batchsize
            shape_top1.update(prec1, shape_score.shape[0])

            shape_features.append(shape_feat.data.cpu())
            shape_labels.append(p_labels)
            shape_scores.append(shape_score.data.cpu())

            batch_time.update(time.time() - end)
            end = time.time()

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
    args = get_args()
    experiment_dir = main(args)
    # os.system('sh run_eval_all.sh 5 %s' % experiment_dir)
