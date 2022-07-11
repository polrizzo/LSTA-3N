import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from colorama import init, Fore, Back
from torch import nn
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torchviz import make_dot

from dataset.tsn import TSNDataSet
from model.module import VideoModel
from utils.average_meter import AverageMeter
from utils.loss import JAN, mmd_rbf, cross_entropy_soft, attentive_entropy
from utils.options import parser
from utils.utils import print_args, save_checkpoint, remove_dummy

import os
import time

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
init(autoreset=True)
gpu_count = torch.cuda.device_count()


def _initialize_lsta(model):
    # Initial setup
    train_params = []
    model.module.lsta_model.train(False)
    for params in model.module.lsta_model.parameters():
        params.requires_grad = False
    for params in model.module.lsta_model.lsta_cell.parameters():
        params.requires_grad = True
        train_params += [params]
    for params in model.module.lsta_model.classifier.parameters():
        params.requires_grad = True
        train_params += [params]
    model.module.lsta_model.to(device)
    model.module.lsta_model.set_loss_fn(nn.CrossEntropyLoss().to(device))
    model.module.lsta_model.set_optimizer_fn(
        torch.optim.Adam(train_params, lr=args.lr_lsta, weight_decay=5e-4, eps=1e-4))
    model.module.lsta_model.set_optim_scheduler(
        torch.optim.lr_scheduler.MultiStepLR(model.module.lsta_model.optimizer_fn, milestones=args.lr_steps_lsta,
                                             gamma=args.lr_decay_lsta))


def _adjust_learning_rate_loss(optimizer, decay, stat_current, stat_previous, op):
    ops = {'>': (lambda x, y: x > y), '<': (lambda x, y: x < y), '>=': (lambda x, y: x >= y),
           '<=': (lambda x, y: x <= y)}
    if ops[op](stat_current, stat_previous):
        for param_group in optimizer.param_groups:
            param_group['lr'] /= decay


def _adjust_learning_rate(optimizer, decay):
    """Sets the learning rate to the initial LR decayed by 10 """
    for param_group in optimizer.param_groups:
        param_group['lr'] /= decay


def _adjust_learning_rate_dann(optimizer, p):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr / (1. + 10 * p) ** 0.75


def _accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    global args, writer_train, writer_val, device
    args = parser.parse_args()

    # === Initialize the training setups === #
    print_args(args, Fore)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(Fore.GREEN + 'Device in using:', str(device))
    start_epoch = 1
    best_prec1 = 0
    cudnn.benchmark = True

    num_class_str = args.num_class.split(",")
    if len(num_class_str) < 1:
        raise Exception("Must specify a number of classes to train")
    else:
        num_class = []
        for num in num_class_str:
            num_class.append(int(num))

    # check the experiments folder existence
    path_exp = args.exp_path + args.modality + '/'
    if not os.path.isdir(path_exp):
        os.makedirs(path_exp)

    if args.tensorboard:
        writer_train = SummaryWriter(path_exp + '/tensorboard_train')
        writer_val = SummaryWriter(path_exp + '/tensorboard_val')

    # === Initialize the model ===#
    print(Fore.CYAN + '---Preparing the model---')
    model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
                       train_segments=args.num_segments, val_segments=args.val_segments,
                       base_model=args.arch, path_pretrained=args.pretrained,
                       add_fc=args.add_fc, fc_dim=args.fc_dim,
                       dropout_i=args.dropout_i, dropout_v=args.dropout_v, partial_bn=not args.no_partialbn,
                       use_bn=args.use_bn if args.use_target != 'none' else 'none',
                       ens_DA=args.ens_DA if args.use_target != 'none' else 'none',
                       n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
                       use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
                       verbose=args.verbose, share_params=args.share_params, mem_size=args.mem_size,
                       outpool_size=args.outPool_size, use_lsta=True if args.use_lsta == 'Y' else False)

    model = torch.nn.DataParallel(model, args.gpus).to(device)

    # check the optimizer
    optimizer = None
    if args.optimizer == 'SGD':
        print(Fore.YELLOW + 'using SGD')
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optimizer == 'Adam':
        print(Fore.YELLOW + 'using Adam')
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        print(Back.RED + 'optimizer not support or specified!!!')
        exit()

    # check the checkpoint
    print(Fore.CYAN + '---Checking the checkpoint---')
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch'])))
            if args.resume_hp:
                print("=> loaded checkpoint hyper-parameters")
                optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(Back.RED + "=> no checkpoint found at '{}'".format(args.resume))
    else:
        print('No checkpoint selected')

    # open log files
    val_best_file = open(path_exp + 'best_val_new.txt', 'a')
    if args.resume:
        train_file = open(path_exp + 'train.log', 'a')
        train_short_file = open(path_exp + 'train_short.log', 'a')
        val_file = open(path_exp + 'val.log', 'a')
    else:
        train_short_file = open(path_exp + 'train_short.log', 'w')
        train_file = open(path_exp + 'train.log', 'w')
        val_file = open(path_exp + 'val.log', 'w')

    # === Data loading ===#
    print(Fore.CYAN + '---Loading data---')
    if args.use_opencv:
        print("Use opencv functions")
    data_length = 1

    # some statistical numbers
    num_source = len(pd.read_pickle(args.train_source_list).index)
    num_target = len(pd.read_pickle(args.train_target_list).index)
    num_iter_source = num_source / args.batch_size[0]
    num_iter_target = num_target / args.batch_size[1]
    num_max_iter = max(num_iter_source, num_iter_target)
    num_source_train = round(num_max_iter * args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
    num_target_train = round(num_max_iter * args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

    # source
    train_source_list = Path(args.train_source_list)
    train_source_data = Path(args.train_source_data)
    source_set = TSNDataSet(train_source_data, train_source_list,
                            num_dataload=num_source_train,
                            num_segments=args.num_segments,
                            total_segments=5,
                            new_length=data_length, modality=args.modality,
                            image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
                                                                            "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                            random_shift=False,
                            test_mode=True,
                            use_spatial_features=args.use_spatial_features
                            )
    source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False,
                                                sampler=source_sampler, num_workers=args.workers, pin_memory=True)
    print('Source length:', '{} samples'.format(len(source_set)))

    # target
    train_target_list = Path(args.train_target_list)
    train_target_data = Path(args.train_target_data)
    target_set = TSNDataSet(train_target_data, train_target_list,
                            num_dataload=num_target_train,
                            num_segments=args.num_segments,
                            total_segments=5,
                            new_length=data_length, modality=args.modality,
                            image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
                                                                            "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                            random_shift=False,
                            test_mode=True,
                            use_spatial_features=args.use_spatial_features
                            )
    target_sampler = torch.utils.data.sampler.RandomSampler(target_set)
    target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False,
                                                sampler=target_sampler, num_workers=args.workers, pin_memory=True)
    print('Target length:', '{} samples'.format(len(target_set)))

    # Optimizer
    criterion = None
    criterion_domain = None
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().to(device)
        criterion_domain = torch.nn.CrossEntropyLoss().to(device)
    else:
        raise ValueError("Unknown loss type")

    # === Training ===#
    print(Fore.CYAN + '---Start training---')
    start_train = time.time()
    loss_c_current = 999  # random large number
    loss_c_previous = 999  # random large number
    beta = args.beta
    gamma = args.gamma
    mu = args.mu

    attn_source_all = torch.Tensor()
    attn_target_all = torch.Tensor()

    if args.use_lsta == 'Y':
        _initialize_lsta(model)

    for epoch in range(start_epoch, args.epochs + 1):
        print(Fore.GREEN + '-Epoch', '{}'.format(epoch))
        print(Fore.GREEN + '-Learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

        # schedule for parameters and learning rate
        alpha = 2 / (1 + math.exp(-1 * epoch / args.epochs)) - 1 if args.alpha < 0 else args.alpha
        if args.lr_adaptive == 'loss':
            _adjust_learning_rate_loss(optimizer, args.lr_decay, loss_c_current, loss_c_previous, '>')
        elif args.lr_adaptive == 'none' and epoch in args.lr_steps:
            _adjust_learning_rate(optimizer, args.lr_decay)

        loss_c, attn_epoch_source, attn_epoch_target, out_mean = train(source_loader, target_loader, model, criterion,
                                                             criterion_domain, optimizer,
                                                             epoch, train_file, train_short_file, alpha, beta, gamma,
                                                             mu)

        # draw the PyTorch execution graphs and traces
        if args.draw_execution_graphs and epoch == 1:
            print(Fore.CYAN + 'Drawing execution graphs and traces')
            make_dot(out_mean, params=dict(model.named_parameters())).render(path_exp + "execution_graph", format="png")

        if args.save_attention >= 0:
            attn_source_all = torch.cat((attn_source_all, attn_epoch_source.unsqueeze(0)))  # save the attention values
            attn_target_all = torch.cat((attn_target_all, attn_epoch_target.unsqueeze(0)))  # save the attention values

        # update the recorded loss_c
        loss_c_previous = loss_c_current
        loss_c_current = loss_c

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            if target_set.labels_available:
                prec1_verb_val = validate(target_loader, model, criterion, num_class, epoch,
                                          log=val_file, tensor_writer=writer_val)
                print(Fore.YELLOW + 'Precision on verb:', prec1_verb_val)

                prec1 = 0
                if args.train_metric == "all":
                    raise Exception("Metric to train not yet implemented")
                elif args.train_metric == "noun":
                    raise Exception("Metric to train not yet implemented")
                elif args.train_metric == "verb":
                    prec1 = prec1_verb_val
                else:
                    raise Exception("invalid metric to train")

                epoch_robustness = epoch / args.epochs
                epoch_robustness_bool = epoch_robustness >= (3 / 10)
                if epoch_robustness == (3 / 10):
                    best_prec1 = 0
                is_best = prec1 > best_prec1
                if is_best and epoch_robustness_bool:
                    best_prec1 = prec1

                line_update = '\n--> updating the best accuracy' if is_best else ''
                line_best = "Best score {} vs Current score {}".format(best_prec1, prec1) + line_update
                print(Fore.RED + line_best)

                best_prec1 = max(prec1, best_prec1)
                if args.tensorboard:
                    writer_val.add_text('Best_Accuracy', str(best_prec1), epoch)
                if args.save_model:
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_prec1': best_prec1,
                        'prec1': prec1,
                    }, is_best, path_exp)
            else:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': 0.0,
                    'prec1': 0.0,
                }, False, path_exp)

    end_train = time.time()
    print(Fore.YELLOW + 'Total training time:', end_train - start_train)
    line_time = 'Total time: {:.3f} '.format(end_train - start_train)

    # close log file
    train_file.close()
    train_short_file.close()
    if target_set.labels_available:
        val_best_file.write('%.3f\n' % best_prec1)
        val_file.write(line_time)
        val_file.close()
    if args.tensorboard:
        writer_train.close()
        writer_val.close()

    if args.save_attention >= 0:
        np.savetxt('attn_source_' + str(args.save_attention) + '.log', attn_source_all.to(device).
                    detach().numpy(), fmt="%s")
        np.savetxt('attn_target_' + str(args.save_attention) + '.log', attn_target_all.to(device).
                    detach().numpy(), fmt="%s")


def _use_pretrained_source_data(source_data, target_data, model, criterion, optimizer, mu, beta, batch_source,
                                label_source_verb):
    # forward pass data again
    _, out_source, out_source_2, _, _, _, _, _, _, _ = model(device, source_data, target_data, beta, mu, is_train=True,
                                                             reverse=False,
                                                             use_spatial_features=args.use_spatial_features)
    # ignore dummy tensors
    out_source_verb = out_source[0][:batch_source]
    out_source_noun = out_source[1][:batch_source]
    out_source_2 = out_source_2[:batch_source]

    # calculate the loss function
    out_verb = out_source_verb.to(device)
    label_verb = label_source_verb.to(device)

    loss_verb = criterion(out_verb, label_verb)
    if args.train_metric == "all":
        loss = 0.5 * loss_verb  # (loss_verb + loss_noun)
    elif args.train_metric == "noun":
        raise Exception('noun was temporarily disable')
    elif args.train_metric == "verb":
        loss = loss_verb  # 0.5*(loss_verb+loss_noun)
    else:
        raise Exception("invalid metric to train")

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()

    if args.clip_gradient is not None:
        total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
        if total_norm > args.clip_gradient and args.verbose:
            print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))
    optimizer.step()


def _print_training_line(params, log):
    line = 'Train: [{0}][{1}/{2}], lr: {lr:.5f}\t' + \
           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' + \
           'Prec@1 {top1_verb.val:.3f} ({top1_verb.avg:.3f})\t' + \
           'Prec@5 {top5_verb.val:.3f} ({top5_verb.avg:.3f})\t' + \
           'Loss {loss.val:.4f} ({loss.avg:.4f})   loss_verb {loss_verb.avg:.4f}\t'
    if args.dis_DA != 'none' and args.use_target != 'none':
        line += 'alpha {alpha:.3f}  loss_d {loss_d.avg:.4f}\t'
    if args.adv_DA != 'none' and args.use_target != 'none':
        line += 'beta {beta[0]:.3f}, {beta[1]:.3f}, {beta[2]:.3f}  loss_a {loss_a.avg:.4f}\t'
    if args.add_loss_DA != 'none' and args.use_target != 'none':
        line += 'gamma {gamma:.6f}  loss_e_verb {loss_e_verb.avg:.4f}\t'
    if args.ens_DA != 'none' and args.use_target != 'none':
        line += 'mu {mu:.6f}  loss_s {loss_s.avg:.4f}\t'

    line = line.format(
        params["epoch"], params["iteration"], params['len_dataset'], batch_time=params['batch_time'],
        data_time=params['data_time'], alpha=params['alpha'], beta=params['beta_new'],
        gamma=params['gamma'], mu=params['mu'],
        loss=params['losses'], loss_verb=params['losses_c_verb'],
        loss_d=params['losses_d'], loss_a=params['losses_a'],
        loss_e_verb=params['losses_e_verb'],
        loss_s=params['losses_s'],
        top1_verb=params['top1_verb'],
        top5_verb=params['top5_verb'],
        lr=params['lr'])
    if params["iteration"] % args.show_freq == 0:
        print(line)
    log.write('%s\n' % line)


def train(source_loader, target_loader, model, criterion, criterion_domain, optimizer, epoch, log,
          log_short, alpha, beta, gamma, mu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_a = AverageMeter()  # adversarial loss
    losses_d = AverageMeter()  # discrepancy loss
    losses_e_verb = AverageMeter()
    losses_s = AverageMeter()  # ensemble loss
    losses_c = AverageMeter()
    losses_c_verb = AverageMeter()  # classification loss
    losses = AverageMeter()
    top1_verb = AverageMeter()
    top5_verb = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    if args.use_lsta == 'Y':
        model.module.lsta_model.optim_scheduler.step()
    model.train()

    end = time.time()
    data_loader = enumerate(zip(source_loader, target_loader))

    # step info
    start_steps = epoch * len(source_loader)
    total_steps = args.epochs * len(source_loader)

    # initialize the embedding
    if args.tensorboard:
        feat_source_display = None
        feat_source_display_verb = None
        label_source_verb_display = None
        label_source_domain_display = None

        feat_target_display = None
        feat_target_display_verb = None
        label_target_verb_display = None
        label_target_domain_display = None

    attn_epoch_source = torch.Tensor()
    attn_epoch_target = torch.Tensor()
    out_verb = torch.Tensor()
    for i, ((source_data, source_label, source_id), (target_data, target_label, target_id)) in data_loader:
        if source_data.size(0) != source_label.size(0) or target_data.size(0) != target_label.size(0):
            print('Source - Skipped for different size: {} {}'.format(source_data.size(0), source_label.size(0)))
            print('Target - Skipped for different size: {} {}'.format(target_data.size(0), target_label.size(0)))
            continue

        # setup hyper parameters
        p = float(i + start_steps) / total_steps
        beta_dann = 2. / (1. + np.exp(-1.0 * p)) - 1
        beta = [beta_dann if beta[i] < 0 else beta[i] for i in
                range(len(beta))]  # replace the default beta if value < 0
        if args.dann_warmup:
            beta_new = [beta_dann * beta[i] for i in range(len(beta))]
        else:
            beta_new = beta
        source_size_ori = source_data.size()  # original shape
        target_size_ori = target_data.size()  # original shape
        batch_source_ori = source_size_ori[0]
        batch_target_ori = target_size_ori[0]

        # add dummy tensors to keep the same batch size for each epoch (for the last epoch)
        if batch_source_ori < args.batch_size[0]:
            if args.use_spatial_features == 'Y':
                source_data_dummy = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1],
                                                source_size_ori[2], source_size_ori[3], source_size_ori[4])
            else:
                source_data_dummy = torch.zeros(args.batch_size[0] - batch_source_ori, source_size_ori[1],
                                                source_size_ori[2])
            source_data = torch.cat((source_data, source_data_dummy))
        if batch_target_ori < args.batch_size[1]:
            if args.use_spatial_features == 'Y':
                target_data_dummy = torch.zeros(args.batch_size[1] - batch_target_ori, target_size_ori[1],
                                                target_size_ori[2], target_size_ori[3], target_size_ori[4])
            else:
                target_data_dummy = torch.zeros(args.batch_size[1] - batch_target_ori, target_size_ori[1],
                                                target_size_ori[2])
            target_data = torch.cat((target_data, target_data_dummy))

        # add dummy tensors to make sure batch size can be divided by gpu #
        if gpu_count != 0 and source_data.size(0) % gpu_count != 0:
            if args.use_spatial_features == 'Y':
                source_data_dummy = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1),
                                                source_data.size(2), source_data.size(3), source_data.size(4))
            else:
                source_data_dummy = torch.zeros(gpu_count - source_data.size(0) % gpu_count, source_data.size(1),
                                                source_data.size(2))
            source_data = torch.cat((source_data, source_data_dummy))
        if gpu_count != 0 and target_data.size(0) % gpu_count != 0:
            if args.use_spatial_features == 'Y':
                target_data_dummy = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1),
                                                target_data.size(2), target_data.size(3), target_data.size(4))
            else:
                target_data_dummy = torch.zeros(gpu_count - target_data.size(0) % gpu_count, target_data.size(1),
                                                target_data.size(2))
            target_data = torch.cat((target_data, target_data_dummy))

        data_time.update(time.time() - end)
        source_label_verb = source_label.to(device)
        target_label_verb = target_label.to(device)

        source_label_verb_frame = None
        target_label_verb_frame = None
        if args.baseline_type == 'frame':
            # expand the size for all the frames
            source_label_verb_frame = source_label_verb.unsqueeze(1).repeat(1, args.num_segments).view(-1)
            target_label_verb_frame = target_label_verb.unsqueeze(1).repeat(1, args.num_segments).view(-1)

        # determine the label for calculating the loss function
        label_source_verb = source_label_verb_frame if args.baseline_type == 'frame' else source_label_verb
        label_target_verb = target_label_verb_frame if args.baseline_type == 'frame' else target_label_verb

        # === Using pre-train source data ===#
        if args.pretrain_source:
            _use_pretrained_source_data(source_data, target_data, model, criterion, optimizer, mu, beta_new
                                        , label_source_verb)

        # === Forward pass data ===#
        attn_source, out_source, out_source_2, pred_domain_source, feat_source, attn_target, out_target, out_target_2, pred_domain_target, feat_target = model(
            device, source_data.to(device), target_data.to(device), beta_new, mu, is_train=True,
            reverse=False, use_spatial_features=args.use_spatial_features)
        # ignore dummy tensors
        attn_source, out_source, out_source_2, pred_domain_source, feat_source = remove_dummy(attn_source, out_source,
                                                                                              out_source_2,
                                                                                              pred_domain_source,
                                                                                              feat_source,
                                                                                              batch_source_ori)
        attn_target, out_target, out_target_2, pred_domain_target, feat_target = remove_dummy(attn_target, out_target,
                                                                                              out_target_2,
                                                                                              pred_domain_target,
                                                                                              feat_target,
                                                                                              batch_target_ori)

        # store the embedding
        if args.tensorboard:
            feat_source_display_verb = feat_source[1] if i == 0 else torch.cat(
                (feat_source_display_verb, feat_source[1]), 0)
            feat_source_display = feat_source[2] if i == 0 else torch.cat((feat_source_display, feat_source[2]), 0)

            label_source_verb_display = label_source_verb if i == 0 else torch.cat(
                (label_source_verb_display, label_source_verb), 0)
            label_source_domain_display = torch.zeros(label_source_verb.size(0)) if i == 0 else torch.cat(
                (label_source_domain_display, torch.zeros(label_source_verb.size(0))), 0)

            feat_target_display_noun = feat_target[0] if i == 0 else torch.cat(
                (feat_target_display_noun, feat_target[0]), 0)
            feat_target_display_verb = feat_target[1] if i == 0 else torch.cat(
                (feat_target_display_verb, feat_target[1]), 0)
            feat_target_display = feat_target[2] if i == 0 else torch.cat((feat_target_display, feat_target[2]), 0)

            label_target_verb_display = label_target_verb if i == 0 else torch.cat(
                (label_target_verb_display, label_target_verb), 0)
            label_target_domain_display = torch.ones(label_target_verb.size(0)) if i == 0 else torch.cat(
                (label_target_domain_display, torch.ones(label_target_verb.size(0))), 0)

        # === Calculate the loss function ===#
        out_verb = out_source[0].to(device)
        label_verb = label_source_verb.to(device)
        loss_verb = criterion(out_verb, label_verb)
        if args.train_metric == "all":
            loss_classification = 0.5 * loss_verb  # (loss_verb + loss_noun)
        elif args.train_metric == "noun":
            raise Exception('noun was temporarily disable')
        elif args.train_metric == "verb":
            loss_classification = loss_verb  # 0.5*(loss_verb+loss_noun)
        else:
            raise Exception("invalid metric to train")

        losses_c_verb.update(loss_verb.item(), out_verb.size(0))
        loss = loss_classification
        losses_c.update(loss_classification.item(), out_verb.size(0))

        # calculate the loss for DA
        # (I) discrepancy-based approach: discrepancy loss
        if args.dis_DA != 'none' and args.use_target != 'none':
            loss_discrepancy = 0

            kernel_muls = [2.0] * 2
            kernel_nums = [2, 5]
            fix_sigma_list = [None] * 2

            if args.dis_DA == 'JAN':
                # ignore the features from shared layers
                feat_source_sel = feat_source[:-args.add_fc]
                feat_target_sel = feat_target[:-args.add_fc]

                size_loss = min(feat_source_sel[0].size(0), feat_target_sel[0].size(0))  # choose the smaller number
                feat_source_sel = [feat[:size_loss] for feat in feat_source_sel]
                feat_target_sel = [feat[:size_loss] for feat in feat_target_sel]

                loss_discrepancy += JAN(feat_source_sel, feat_target_sel, kernel_muls=kernel_muls,
                                        kernel_nums=kernel_nums, fix_sigma_list=fix_sigma_list, ver=2)

            else:
                # extend the parameter list for shared layers
                kernel_muls.extend([kernel_muls[-1]] * args.add_fc)
                kernel_nums.extend([kernel_nums[-1]] * args.add_fc)
                fix_sigma_list.extend([fix_sigma_list[-1]] * args.add_fc)

                for l in range(0,
                               args.add_fc + 2):  # loss from all the features (+2 because of frame-aggregation layer + final fc layer)
                    if args.place_dis[l] == 'Y':
                        # select the data for calculating the loss (make sure source # == target #)
                        size_loss = min(feat_source[l].size(0), feat_target[l].size(0))  # choose the smaller number
                        # select
                        feat_source_sel = feat_source[l][:size_loss]
                        feat_target_sel = feat_target[l][:size_loss]

                        # break into multiple batches to avoid "out of memory" issue
                        size_batch = min(256, feat_source_sel.size(0))
                        feat_source_sel = feat_source_sel.view((-1, size_batch) + feat_source_sel.size()[1:])
                        feat_target_sel = feat_target_sel.view((-1, size_batch) + feat_target_sel.size()[1:])

                        if args.dis_DA == 'DAN':
                            losses_mmd = [mmd_rbf(feat_source_sel[t], feat_target_sel[t], kernel_mul=kernel_muls[l],
                                                  kernel_num=kernel_nums[l], fix_sigma=fix_sigma_list[l], ver=2) for t
                                          in range(feat_source_sel.size(0))]
                            loss_mmd = sum(losses_mmd) / len(losses_mmd)

                            loss_discrepancy += loss_mmd
                        else:
                            raise NameError('not in dis_DA!!!')

            losses_d.update(loss_discrepancy.item(), feat_source[0].size(0))
            loss += alpha * loss_discrepancy

        # (II) adversarial discriminative model: adversarial loss
        if args.adv_DA != 'none' and args.use_target != 'none':
            loss_adversarial = 0
            pred_domain_all = []
            pred_domain_target_all = []

            for l in range(len(args.place_adv)):
                if args.place_adv[l] == 'Y':
                    # reshape the features (e.g. 128x5x2 --> 640x2)
                    pred_domain_source_single = pred_domain_source[l].view(-1, pred_domain_source[l].size()[-1])
                    pred_domain_target_single = pred_domain_target[l].view(-1, pred_domain_target[l].size()[-1])

                    # prepare domain labels
                    source_domain_label = torch.zeros(pred_domain_source_single.size(0)).long()
                    target_domain_label = torch.ones(pred_domain_target_single.size(0)).long()
                    domain_label = torch.cat((source_domain_label, target_domain_label), 0)

                    domain_label = domain_label.to(device)

                    pred_domain = torch.cat((pred_domain_source_single, pred_domain_target_single), 0)
                    pred_domain_all.append(pred_domain)
                    pred_domain_target_all.append(pred_domain_target_single)

                    if args.pred_normalize == 'Y':  # use the uncertainly method (in construction......)
                        pred_domain = pred_domain / pred_domain.var().log()
                    loss_adversarial_single = criterion_domain(pred_domain, domain_label)

                    loss_adversarial += loss_adversarial_single

            losses_a.update(loss_adversarial.item(), pred_domain.size(0))
            loss += loss_adversarial

        # (III) other loss
        # entropy loss for target data
        if args.add_loss_DA == 'target_entropy' and args.use_target != 'none':
            loss_entropy_verb = cross_entropy_soft(out_target[0])
            losses_e_verb.update(loss_entropy_verb.item(), out_target[0].size(0))
            if args.train_metric == "all":
                loss += gamma * 0.5 * loss_entropy_verb  # (loss_entropy_verb + loss_entropy_noun)
            elif args.train_metric == "noun":
                raise Exception('noun was temporarily disable')
            elif args.train_metric == "verb":
                loss += gamma * loss_entropy_verb
            else:
                raise Exception("invalid metric to train")
        # attentive entropy loss
        if args.add_loss_DA == 'attentive_entropy' and args.use_attn != 'none' and args.use_target != 'none':
            loss_entropy_verb = attentive_entropy(torch.cat((out_verb, out_target[0]), 0), pred_domain_all[1])
            losses_e_verb.update(loss_entropy_verb.item(), out_target[0].size(0))
            if args.train_metric == "all":
                loss += gamma * 0.5 * loss_entropy_verb  # (loss_entropy_verb + loss_entropy_noun)
            elif args.train_metric == "noun":
                raise Exception('noun was temporarily disable')
            elif args.train_metric == "verb":
                loss += gamma * loss_entropy_verb
            else:
                raise Exception("invalid metric to train")

        pred_verb = out_verb
        prec1_verb, prec5_verb = _accuracy(pred_verb.data, label_verb, topk=(1, 5))

        losses.update(loss.item())
        top1_verb.update(prec1_verb.item(), out_verb.size(0))
        top5_verb.update(prec5_verb.item(), out_verb.size(0))

        # compute gradient and do SGD step (LSTA is already included)
        optimizer.zero_grad()
        loss.backward()
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient and args.verbose:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()
        if args.use_lsta == 'Y':
            model.module.lsta_model.optimizer_fn.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            _print_training_line(
                {
                    "epoch": epoch,
                    "iteration": i,
                    "len_dataset": len(source_data),
                    "gamma": gamma,
                    "mu": mu,
                    "batch_time": batch_time,
                    "data_time": data_time,
                    "alpha": alpha,
                    "beta_new": beta_new,
                    "losses": losses,
                    "losses_c_verb": losses_c_verb,
                    "losses_d": losses_d,
                    "losses_a": losses_a,
                    "losses_e_verb": losses_e_verb,
                    "losses_s": losses_s,
                    "top1_verb": top1_verb,
                    "top5_verb": top5_verb,
                    "lr": optimizer.param_groups[0]['lr']
                }, log)

        # adjust the learning rate for ech step (e.g. DANN)
        if args.lr_adaptive == 'dann':
            _adjust_learning_rate_dann(optimizer, p)

        # save attention values w/ the selected class
        if args.save_attention >= 0:
            attn_source = attn_source[source_label == args.save_attention]
            attn_target = attn_target[target_label == args.save_attention]
            attn_epoch_source = torch.cat((attn_epoch_source, attn_source.to(device)))
            attn_epoch_target = torch.cat((attn_epoch_target, attn_target.to(device)))

    # update the embedding every epoch
    if args.tensorboard:
        writer_train.add_scalar("loss/verb", losses_c_verb.avg, epoch)
        writer_train.add_scalar("acc/verb", top1_verb.avg, epoch)
        if args.adv_DA != 'none' and args.use_target != 'none':
            writer_train.add_scalar("loss/domain", loss_adversarial, epoch)

    return losses_c.avg, attn_epoch_source.mean(0), attn_epoch_target.mean(0), out_verb.mean()


def validate(target_loader, model, criterion, num_class, epoch, log=None, tensor_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_verb = AverageMeter()
    top5_verb = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    # initialize the embedding
    if args.tensorboard:
        feat_val_display = None
        label_val_verb_display = None

    for i, (val_data, val_label, _) in enumerate(target_loader):
        val_size_ori = val_data.size()  # original shape
        batch_val_ori = val_size_ori[0]

        # add dummy tensors to keep the same batch size for each epoch (for the last epoch)
        if batch_val_ori < args.batch_size[2]:
            if args.use_spatial_features == 'Y':
                val_data_dummy = torch.zeros(args.batch_size[2] - batch_val_ori, val_size_ori[1], val_size_ori[2],
                                             val_size_ori[3], val_size_ori[4])
            else:
                val_data_dummy = torch.zeros(args.batch_size[2] - batch_val_ori, val_size_ori[1], val_size_ori[2])
            val_data = torch.cat((val_data, val_data_dummy))
        # add dummy tensors to make sure batch size can be divided by gpu #
        if gpu_count != 0 and val_data.size(0) % gpu_count != 0:
            if args.use_spatial_features == 'Y':
                val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1),
                                             val_data.size(2), val_data.size(3), val_data.size(4))
            else:
                val_data_dummy = torch.zeros(gpu_count - val_data.size(0) % gpu_count, val_data.size(1),
                                             val_data.size(2))
            val_data = torch.cat((val_data, val_data_dummy))

        val_label_verb = val_label.to(device)
        with torch.no_grad():
            if args.baseline_type == 'frame':
                # expand the size for all the frames
                val_label_verb_frame = val_label_verb.unsqueeze(1).repeat(1, args.num_segments).view(-1)

            # compute output
            _, _, _, _, _, attn_val, out_val, out_val_2, pred_domain_val, feat_val = model(device, val_data, val_data,
                                                                                           [0] * len(args.beta), 0,
                                                                                           is_train=False,
                                                                                           reverse=False,
                                                                                           use_spatial_features=args.use_spatial_features)
            # ignore dummy tensors
            attn_val, out_val, out_val_2, pred_domain_val, feat_val = remove_dummy(attn_val, out_val, out_val_2,
                                                                                   pred_domain_val, feat_val,
                                                                                   batch_val_ori)
            # measure accuracy and record loss
            label_verb = val_label_verb_frame if args.baseline_type == 'frame' else val_label_verb
            # store the embedding
            if args.tensorboard:
                feat_val_display = feat_val[1] if i == 0 else torch.cat((feat_val_display, feat_val[1]), 0)
                label_val_verb_display = label_verb if i == 0 else torch.cat((label_val_verb_display, label_verb), 0)

            pred_verb = out_val[0]
            if args.baseline_type == 'tsn':
                pred_verb = pred_verb.view(val_label.size(0), -1, num_class).mean(
                    dim=1)  # average all the segments (needed when num_segments != val_segments)
            loss_verb = criterion(pred_verb, label_verb)
            if args.train_metric == "all":
                loss = 0.5 * loss_verb  # * (loss_verb + loss_noun)
            elif args.train_metric == "noun":
                raise Exception('noun is temporally unavaiable')
            elif args.train_metric == "verb":
                loss = loss_verb  # 0.5*(loss_verb+loss_noun)
            else:
                raise Exception("invalid metric to train")
            prec1_verb, prec5_verb = _accuracy(pred_verb.data, label_verb, topk=(1, 5))

            losses.update(loss.item(), out_val[0].size(0))
            top1_verb.update(prec1_verb.item(), out_val[0].size(0))
            top5_verb.update(prec5_verb.item(), out_val[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                line = 'Test: [{0}][{1}/{2}]\t' + \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' + \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t' + \
                       'Prec@1 verb {top1_verb.val:.3f} ({top1_verb.avg:.3f})\t' + \
                       'Prec@5 verb {top5_verb.val:.3f} ({top5_verb.avg:.3f})\t'

                line = line.format(
                    epoch, i, len(target_loader), batch_time=batch_time, loss=losses,
                    top1_verb=top1_verb, top5_verb=top5_verb)
                if i % args.show_freq == 0:
                    print(line)
                if log is not None:
                    log.write('%s\n' % line)

    if args.tensorboard and tensor_writer is not None:  # update the embedding every iteration
        tensor_writer.add_scalar("acc/verb", top1_verb.avg, epoch)
        if epoch == 20:
            tensor_writer.add_embedding(feat_val_display, metadata=label_val_verb_display.data, global_step=epoch,
                                        tag='validation')
    return top1_verb.avg


if __name__ == "__main__":
    main()
