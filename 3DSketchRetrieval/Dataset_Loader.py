import torch.utils.data
import torch
from config import DATASETS, NUM_VIEWS

def get_dataloader(args):
    list_file = args.list_file
    if args.name == 'pointnet':
        from dataset.PointCloudLoader import PointCloudDataLoader
        if args.network_target:
            train_shape_dataset = PointCloudDataLoader(args, list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='train', data_type='network')
            test_shape_dataset = PointCloudDataLoader(args, list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='val', data_type='network')
        else:
            train_shape_dataset = PointCloudDataLoader(args, list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='train', data_type='shape')
            test_shape_dataset = PointCloudDataLoader(args, list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='val', data_type='shape')

        train_sketch_dataset = PointCloudDataLoader(args, list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='train', data_type='sketch', abstract=args.abstract, target=args.sketch_target, sketch_dir=args.sketch_dir, random_sample=args.random_sample)
        test_sketch_dataset = PointCloudDataLoader(args, list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='val', data_type='sketch', abstract=args.abstract, sketch_dir=args.sketch_dir, random_sample=args.random_sample)
    elif args.name == 'ngvnn':
        from dataset.MultiViewLoader import MultiviewImgDataset

        train_shape_dataset = MultiviewImgDataset(set='train', list_file=list_file, data_type='shape', view_type=args.view_type)
        train_sketch_dataset = MultiviewImgDataset(set='train', list_file=list_file, data_type='sketch', view_type=args.view_type)
        test_shape_dataset = MultiviewImgDataset(set='val', list_file=list_file, data_type='shape',view_type=args.view_type)
        test_sketch_dataset = MultiviewImgDataset(set='val', list_file=list_file, data_type='sketch', view_type=args.view_type)
    elif args.name == 'sbr':
        from dataset.MultiViewLoader import MultiviewImgDataset
        from dataset.ImageLoader import ImgDataset
        train_shape_dataset = MultiviewImgDataset(set='train', list_file=list_file, data_type='shape', view_type=args.view_type)
        train_sketch_dataset = ImgDataset(set='train', list_file=list_file, test_mode=False, shuffle=False, view_type=args.view_type, abstract=args.abstract)
        test_shape_dataset = MultiviewImgDataset(set='val', list_file=list_file, data_type='shape',view_type=args.view_type)
        test_sketch_dataset = ImgDataset(set='val', list_file=list_file, shuffle=False, view_type=args.view_type, abstract=args.abstract)

    else:
        NotImplementedError

    from dataset.TripletSampler import BalancedBatchSampler
    train_shape_sampler = BalancedBatchSampler(train_shape_dataset.labels, n_classes=args.n_classes, n_samples=args.n_samples, seed=0, n_dataset=len(train_sketch_dataset.labels))
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=train_shape_sampler, num_workers=4)
    train_sketch_sampler = BalancedBatchSampler(train_sketch_dataset.labels, n_classes=args.n_classes, n_samples=args.n_samples, seed=0)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=train_sketch_sampler, num_workers=4)

    test_shape_loader = torch.utils.data.DataLoader(test_shape_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_sketch_loader = torch.utils.data.DataLoader(test_sketch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_shape_loader, train_sketch_loader, test_shape_loader, test_sketch_loader

def get_ngvnn_dataloader(args):
    from dataset.TripletSampler import BalancedBatchSampler
    list_file = DATASETS[args.dataset]['list_file']

    if args.double_type == 'pointnet':
        from dataset.PointCloudLoader import PointCloudDataLoader

        train_shape_dataset = PointCloudDataLoader(list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='train', sketch=False, abstract=args.abstract)
        train_sketch_dataset = PointCloudDataLoader(list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='train', shape=False, abstract=args.abstract)

        test_shape_dataset = PointCloudDataLoader(list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='valid', sketch=False, abstract=args.abstract)
        test_sketch_dataset = PointCloudDataLoader(list_file=list_file, npoint=args.num_point, uniform=args.uniform,
                                             split='valid', shape=False, abstract=args.abstract)
    elif args.double_type == 'ngvnn':
        from dataset.MultiViewLoader import MultiviewImgDataset
        train_shape_dataset = MultiviewImgDataset(set='train', list_file=list_file, scale_aug=False,
                                            rot_aug=False, test_mode=False, num_views=NUM_VIEWS, shuffle=False,
                                            sketch=False, abstract=args.abstract)
        train_sketch_dataset = MultiviewImgDataset(set='train', list_file=list_file, scale_aug=False,
                                            rot_aug=False, test_mode=False, num_views=NUM_VIEWS, shuffle=False,
                                            shape=False, abstract=args.abstract)

        test_shape_dataset = MultiviewImgDataset(set='valid', list_file=list_file, scale_aug=False,
                                           rot_aug=False, num_views=NUM_VIEWS, sketch=False,
                                           abstract=args.abstract)
        test_sketch_dataset = MultiviewImgDataset(set='valid', list_file=list_file, scale_aug=False,
                                           rot_aug=False, num_views=NUM_VIEWS, shape=False,
                                           abstract=args.abstract)
    else:
        from dataset.MultiViewLoader import MultiviewImgDataset
        from dataset.ImageLoader import ImgDataset
        train_shape_dataset = MultiviewImgDataset(set='train', list_file=list_file, scale_aug=False, rot_aug=False,
                                                  test_mode=False, num_views=NUM_VIEWS, shuffle=False, sketch=False)
        train_sketch_dataset = ImgDataset(set='train', list_file=list_file, scale_aug=False, rot_aug=False, test_mode=False,
                                          shuffle=False)
        test_shape_dataset = MultiviewImgDataset(set='valid', list_file=list_file, scale_aug=False, rot_aug=False,
                                                 test_mode=False, num_views=NUM_VIEWS, shuffle=False, sketch=False)
        test_sketch_dataset = ImgDataset(set='valid', list_file=list_file, scale_aug=False, rot_aug=False, shuffle=False)

    train_sketch_sampler = BalancedBatchSampler(train_sketch_dataset.labels, n_classes=args.n_classes,n_samples=args.n_samples)
    train_sketch_loader = torch.utils.data.DataLoader(train_sketch_dataset, batch_sampler=train_sketch_sampler,
                                                      num_workers=2)  # shuffle needs to be false! it's done within the trainer
    train_shape_sampler = BalancedBatchSampler(train_shape_dataset.labels, n_classes=args.n_classes,n_samples=args.n_samples)
    train_shape_loader = torch.utils.data.DataLoader(train_shape_dataset, batch_sampler=train_shape_sampler,
                                                     num_workers=2)  # shuffle needs to be false! it's done within the trainer
    test_shape_sampler = BalancedBatchSampler(test_shape_dataset.labels, n_classes=args.n_classes,n_samples=args.n_samples)
    test_shape_loader = torch.utils.data.DataLoader(test_shape_dataset, batch_sampler=test_shape_sampler,
                                                    num_workers=2)  # shuffle needs to be false! it's done within the trainer
    test_sketch_sampler = BalancedBatchSampler(test_sketch_dataset.labels, n_classes=args.n_classes,n_samples=args.n_samples)
    test_sketch_loader = torch.utils.data.DataLoader(test_sketch_dataset, batch_sampler=test_sketch_sampler,
                                                     num_workers=2)  # shuffle needs to be false! it's done within the trainer

    return train_shape_loader, train_sketch_loader, test_shape_loader, test_sketch_loader
