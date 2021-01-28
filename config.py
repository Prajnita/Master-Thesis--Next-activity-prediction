import os
import argparse
import utils


def load():
    parser = argparse.ArgumentParser()

    # dnn
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--dnn_architecture', default= 2, type=int)

    # all models
    parser.add_argument('--task', default="next_activity")
    parser.add_argument('--learning_rate', default=0.002, type=float)  # dnc 0.0001 #lstm 0.002

    # evaluation
    parser.add_argument('--num_folds', default=10, type=int)  # 10
    parser.add_argument('--cross_validation', default=True, type=utils.str2bool)
    parser.add_argument('--split_rate_test', default=0.5, type=float)  # only if cross validation is deactivated
    parser.add_argument('--batch_size_train', default=256, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # data
    parser.add_argument('--data_set', default="helpdesk_converted_no_context.csv")
    parser.add_argument('--data_dir', default="data/")
    parser.add_argument('--checkpoint_dir', default="./checkpoints/")
    parser.add_argument('--result_dir', default="./results/")

    # gpu processing
    parser.add_argument('--gpu_ratio', default=1.0, type=float)
    parser.add_argument('--cpu_num', default=6, type=int)
    parser.add_argument('--gpu_device', default="0", type=str)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    return args
