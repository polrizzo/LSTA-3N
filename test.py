from pathlib import Path

import pandas as pd
import torch
from colorama import Fore
from tqdm import tqdm

from dataset.tsn import TSNDataSet
from main import validate
from model.module import VideoModel
from utils.options import parser
from utils.utils import print_args

gpu_count = torch.cuda.device_count()


def main():
    global args, device
    args = parser.parse_args()

    # === Initialize the training setups === #
    print_args(args, Fore)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_class_str = args.num_class.split(",")
    if len(num_class_str) < 1:
        raise Exception("Must specify a number of classes to train")
    else:
        num_class = []
        for num in num_class_str:
            num_class.append(int(num))

    criterion = torch.nn.CrossEntropyLoss().to(device)
    print(Fore.CYAN + '---Preparing the model---')
    model = VideoModel(num_class, args.baseline_type, args.frame_aggregation, args.modality,
                       train_segments=args.test_segments if args.baseline_type == 'video' else 1,
                       val_segments=args.test_segments if args.baseline_type == 'video' else 1,
                       base_model=args.arch, add_fc=args.add_fc, fc_dim=args.fc_dim, share_params=args.share_params,
                       dropout_i=args.dropout_i, dropout_v=args.dropout_v, use_bn=args.use_bn, partial_bn=False,
                       n_rnn=args.n_rnn, rnn_cell=args.rnn_cell, n_directions=args.n_directions, n_ts=args.n_ts,
                       use_attn=args.use_attn, n_attn=args.n_attn, use_attn_frame=args.use_attn_frame,
                       verbose=args.verbose, before_softmax=False, mem_size=args.mem_size,
                       outpool_size=args.outPool_size, use_lsta=True if args.use_lsta == 'Y' else False)

    # === Data loading ===#
    print(Fore.CYAN + '---Loading data---')

    data_length = 1 if args.modality == "RGB" else 1
    num_target = len(pd.read_pickle(args.train_target_list).index)
    num_iter_target = num_target / args.batch_size[1]
    num_target_train = round(num_iter_target * args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

    train_target_list = Path(args.train_target_list)
    train_target_data = Path(args.train_target_data)
    data_set = TSNDataSet(train_target_data, train_target_list,
                          num_dataload=num_target_train,
                          num_segments=args.test_segments,
                          total_segments=5,
                          new_length=data_length, modality=args.modality,
                          image_tmpl="img_{:05d}.t7" if args.modality in ["RGB", "RGBDiff", "RGBDiff2",
                                                                          "RGBDiffplus"] else args.flow_prefix + "{}_{:05d}.t7",
                          test_mode=True,
                          use_spatial_features=args.use_spatial_features
                          )
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size[1], shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    data_gen = tqdm(data_loader)
    output = []
    attn_values = torch.Tensor()

    validate(data_loader, model, criterion, num_class, 0, None, None)


if __name__ == "__main__":
    main()
