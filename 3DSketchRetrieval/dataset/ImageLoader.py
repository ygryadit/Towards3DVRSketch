import numpy as np
import glob
import torch.utils.data
import os
import math
from PIL import Image
import torch
from torchvision import transforms, datasets
from tools.misc import load_string_list
from config import VIEW_DIR, ROOT_DIR

class ImgDataset(torch.utils.data.Dataset):

    def __init__(self, set='train', list_file='', view_type='', scale_aug=False, rot_aug=False, test_mode=True,
                 shuffle=False, abstract=0.5):

        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.name_list = load_string_list(list_file.format(set))
        self.file_paths = []
        self.labels = []
        # self.sketch_end = '_sketch_{}_11.png'.format(abstract)
        self.sketch_end = '_sketch_{}_smooth_1.0_12.png'.format(abstract)

        view_dir = os.path.join(VIEW_DIR, 'sketch_'+view_type)

        if abstract <= 1.0:
            for line in self.name_list:
                model_name, class_id = line.split(' ')
                sketch_paths = view_dir+ '/' +  model_name + self.sketch_end
                self.labels.append(int(class_id))
                self.file_paths.append(sketch_paths)
        elif abstract <= 2.0:
            human_list = '/yrfs1/intern/lingluo3/data/list/ModelNet10/equal/human_test.txt'
            name_list = [line.rstrip() for line in open(human_list)]
            # view_dir = os.path.join(VIEW_DIR, 'human_' + view_type)
            view_dir = os.path.join(VIEW_DIR, 'lines')
            for line in name_list:
                model_name, class_id = line.split(' ')
                # sketch_path = view_dir+ '/' + model_name + '_11.png'
                sketch_path = view_dir+ '/' + model_name + '_smooth_1.0_12.png'

                self.file_paths.append(sketch_path)
                self.labels.append(int(class_id))
        else:
            view_dir = os.path.join(VIEW_DIR, 'npr_png')
            self.sketch_end = '_sketch_0.0_11.png'.format(abstract)
            human_list = '/yrfs1/intern/lingluo3/data/list/ModelNet10/equal/human_test.txt'
            name_list = [line.rstrip() for line in open(human_list)]

            for line in name_list:
                model_name, class_id = line.split(' ')
                sketch_paths = view_dir+ '/' +  model_name + self.sketch_end
                self.labels.append(int(class_id))
                self.file_paths.append(sketch_paths)




        print('The size of %s data is %d' % (set, len(self.labels)))

        if shuffle == True:
            # permute
            rand_idx = np.random.permutation(len(self.file_paths) )
            filepaths_new = []
            labels_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.file_paths[i])
                labels_new.extend(self.labels[i])
            self.file_paths = filepaths_new
            self.labels = labels_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        class_id = self.labels[idx]
        # Use PIL instead
        im = Image.open(path).convert('RGB')
        if self.transform:
            im = self.transform(im)

        return (im, class_id)



if __name__ == "__main__":
    from config import DATASETS
    from dataset.TripletSampler import BalancedBatchSampler

    dataset = 'ModelNet10'
    list_file = DATASETS[dataset]['list_file']

    train_dataset = ImgDataset(set='train', list_file=list_file, scale_aug=False, rot_aug=False,
                                        shuffle=False)
    train_dataset = ImgDataset(set='valid', list_file=list_file, scale_aug=False, rot_aug=False,
                                        shuffle=False)
    train_dataset = ImgDataset(set='test', list_file=list_file, scale_aug=False, rot_aug=False,
                                        shuffle=False)

    # train_dataset = ImgDataset(set='valid', list_file=list_file, scale_aug=False, rot_aug=False,
    #                                     shuffle=False)
    # sampler = Sampler(train_dataset)
    # train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=2, n_samples=2)
    # DataLoader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler)

    # DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

    # for labels, images in DataLoader:
    #     print(labels.shape)
    #     print(images.shape)
