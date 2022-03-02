import argparse

def add_args(parser):
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    # data options
    parser.add_argument('-list_file', type=str, default="hs/{}.txt",
                        help="Path to the namelist file")
    parser.add_argument('-num_class', default=10, type=int, help='number of classes')
    parser.add_argument('-data_dir', type=str, default="/vol/vssp/datasets/multiview/3VS/all_networks",
                        help="Path to the training data")

    parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="pointnet")
    parser.add_argument('-abstract', type=str, default="0.5", help='The degree of abstractness')

    parser.add_argument("-pretrain", default=False, dest='pretrain', action='store_true')
    parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11_bn")

    parser.add_argument('-batch_size', type=int, default=8, help='batch size in training [default: 24]')
    parser.add_argument('-model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('-epoch', default=1, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('-learning_rate', default=0.001, type=float,
                        help='learning rate in training [default: 0.001]')
    parser.add_argument('-num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('-optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('-log_dir', type=str, help='experiment root')
    parser.add_argument('-decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('-uniform', action='store_true', default=False,
                        help='Whether to use uniform [default: False]')

    # Triplet Loss
    parser.add_argument('-margin', type=float, default=1.8, help='margin for triplet loss / triplet center loss')
    parser.add_argument('-w1', type=float, default=1, help='weight for classification loss')
    parser.add_argument('-w2', type=float, default=0.1)
    parser.add_argument('-w3', type=float, default=0.1)

    parser.add_argument('-n_classes', type=int, default=4)
    parser.add_argument('-n_samples', type=int, default=1)
    parser.add_argument("-triplet_type", type=str, help="Type of the triplet loss", default="tl", choices={'tl', 'tcl'})

    parser.add_argument('-print-freq', type=int, default=50)
    parser.add_argument('-gradient_clip', type=float, default=0.05)  # previous i set it to be 0.01
    parser.add_argument("-network_target", action='store_true', help="whether to use curve network as reconstruction target", default=False)

    parser.add_argument("-sketch_anchor", action='store_true', help="whether to use sketches as anchors", default=False)
    parser.add_argument("-reconstruct", action='store_true', help="whether to use reconstruction branch", default=False)
    parser.add_argument("-sketch_target", type=str, default='')

    parser.add_argument("-view_type", type=str, default='whiteshaded')
    parser.add_argument("-random_sample", action='store_true', help="whether to use random sampling", default=False)

    parser.add_argument('-debug', action='store_true',
                        help='Whether debug locally.')
    parser.add_argument("-model_path", type=str, default='')

    return parser

def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='3D sketch to 3d shape retrieval Experiment')
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args