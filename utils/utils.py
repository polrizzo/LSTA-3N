import shutil

import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools


def random_select_batch(input_tensor, num):
    id_all = torch.randperm(input_tensor.size(0)).cuda()
    unique_id = id_all[:num]
    return unique_id, input_tensor[unique_id]


def plot_confusion_matrix(path, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    num_classlabels = cm.sum(axis=1)  # count the number of true labels for all the classes
    np.putmask(num_classlabels, num_classlabels == 0, 1)  # avoid zero division

    if normalize:
        cm = cm.astype('float') / num_classlabels[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(13, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    factor = 100 if normalize else 1
    fmt = '.0f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * factor, fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(path)


def print_args(args, color):
    print(color.GREEN + 'Baseline:', args.baseline_type)
    print(color.GREEN + 'Frame aggregation method:', args.frame_aggregation)
    print(color.GREEN + 'Current architecture:', args.arch)
    print(color.GREEN + 'Number of classes:', args.num_class)
    print(color.GREEN + 'Target data usage:', args.use_target)
    print(color.GREEN + 'Use spatial features:', args.use_spatial_features)
    print(color.GREEN + 'Number of workers:', args.workers)
    print(color.GREEN + 'Tensorboard:', args.tensorboard)
    
    if args.use_target == 'none':
        print(color.GREEN + 'No Domain Adaptation')
    else:
        if args.dis_DA != 'none':
            print(color.GREEN + 'Apply the discrepancy-based Domain Adaptation approach:', args.dis_DA)
            if len(args.place_dis) != args.add_fc + 2:
                raise ValueError(color.RED + 'len(place_dis) should be equal to add_fc + 2')

        if args.adv_DA != 'none':
            print(color.GREEN + 'Apply the adversarial-based Domain Adaptation approach:', args.adv_DA)

        if args.use_bn != 'none':
            print(color.GREEN + 'Apply the adaptive normalization approach:', args.use_bn)

    print(color.GREEN + 'Current modality:', args.modality)
    print(color.GREEN + 'From dataset', args.source_domain, color.GREEN + 'to dataset', args.target_domain)

    print("-------------------------------------------------------")
    print(color.GREEN + 'Train source list:', args.train_source_list)
    print(color.GREEN + 'Train target list:', args.train_target_list)
    print(color.GREEN + 'Train source data:', args.train_source_data)
    print(color.GREEN + 'Train target data:', args.train_target_data)
    print(color.GREEN + 'Experiment path:', args.exp_path)
    print("-------------------------------------------------------")


def save_checkpoint(state, is_best, path_exp, filename='checkpoint.pth.tar'):
    path_file = path_exp + filename
    torch.save(state, path_file)
    if is_best:
        path_best = path_exp + 'model_best.pth.tar'
        shutil.copyfile(path_file, path_best)


def remove_dummy(attn, out_1, out_2, pred_domain, feat, batch_size):
    attn = attn[:batch_size]
    if isinstance(out_1, (list, tuple)):
        out_1 = (out_1[0][:batch_size], out_1[1][:batch_size])
    else:
        out_1 = out_1[:batch_size]
    out_2 = out_2[:batch_size]
    pred_domain = [pred[:batch_size] for pred in pred_domain]
    feat = [f[:batch_size] for f in feat]
    return attn, out_1, out_2, pred_domain, feat
