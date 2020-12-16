from Dataset_Loader import get_test_dataloader
import os, json
import importlib
import torch, time
from tools.evaluation import compute_distance, compute_acc_at_k
import numpy as np
def validate(sketch_dataloader, shape_dataloader, model, save=False, save_dir=''):
    sketch_features = []
    shape_features = []
    recon_list = []
    model = model.eval()
    start_time = time.time()

    with torch.no_grad():
        for i, data in enumerate(sketch_dataloader):
            sketch_points = data[0].cuda()
            sketch_points = sketch_points.transpose(2, 1)
            B = sketch_points.shape[0]
            _, sketch_z, recon = model(sketch_points, recon_index=[0, B])
            sketch_features.append(sketch_z.data.cpu())
            recon_list.append(recon.transpose(2, 1).data.cpu())
        # for i, data in enumerate(shape_dataloader):
        #     shape_points = data[0].cuda()
        #     shape_points = shape_points.transpose(2, 1)
        #     shape_z = model(shape_points, train=False)
        #     shape_features.append(shape_z.data.cpu())

    inference_duration = time.time() - start_time
    start_time = time.time()

    # shape_features = torch.cat(shape_features, 0).numpy()
    # sketch_features = torch.cat(sketch_features, 0).numpy()
    recon_list = torch.cat(recon_list, 0).numpy()

    # d_feat_z = compute_distance(sketch_features.copy(), shape_features.copy(), l2=True)
    # acc_at_k_feat_z = compute_acc_at_k(d_feat_z)
    eval_duration = time.time() - start_time

    if save:
        # np.save(os.path.join(save_dir, 'shape_feat_{}.npy'.format(batch_size)), shape_features)
        # np.save(os.path.join(save_dir, 'sketch_feat_{}.npy'.format(batch_size)), sketch_features)
        # np.save(os.path.join(save_dir, 'd_feat.npy'), d_feat_z)
        np.save(os.path.join(save_dir, 'recon.npy'), recon_list)


    print(
        "Inference Time [%3.2fs]  Eval Time [%3.2fs]"
        % (inference_duration, eval_duration))

    # for acc_z_i, k in zip(acc_at_k_feat_z, [1, 5, 10]):
    #     print(' * Acc@{:d} z acc {:.4f}'.format(k, acc_z_i))

    # return acc_at_k_feat_z

def main(args):
    model_path = args.model_path
    assert os.path.exists(model_path), "Model path doesn't exist!"

    config_f = os.path.join(model_path, 'logs/config.json')
    with open(config_f, 'r') as f:
        train_args = json.load(f)
        print(train_args)

    test_shape_loader, test_sketch_loader = get_test_dataloader(args)
    model_name = 'pointnet2_cls_msg'  # os.listdir(model_path + '/logs')[1].split('.')[0]
    model = importlib.import_module('models.' + model_name)

    num_class = 10
    classifier = model.get_model(num_class, num_points=train_args['num_point'], decoder=train_args['reconstruct']).cuda().eval()

    checkpoint = torch.load(model_path + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    validate(test_sketch_loader, test_shape_loader, classifier, save=True, save_dir=model_path)

if __name__ == '__main__':
    from args import get_args
    args = get_args()
    if args.windows:
        from args import get_parser
        parser = get_parser()
        args = parser.parse_args(args=[
            # '-debug',
            '-epoch', '100',
            '-margin', '1.8',
            '-abstract', '1.0',
            '-uniform',
            '-reconstruct',
            '-sketch_target', 'network',
            '-model_path', r'C:\Users\ll00931\PycharmProjects\Towards3DVRSketch\3DSketchRetrieval\save\pointnet\2020-11-20_00-16_UWS62899',
            '-list_file', 'hs/{}.txt',
            '-data_dir', r'C:\Users\ll00931\Documents\chair_1005\all_networks'
            ])
    experiment_dir = main(args)
