import numpy as np
import torch.utils.data
from PIL import Image
import torch
from torchvision import transforms, datasets
from tools.misc import load_string_list
from config import VIEW_DIR
import os

class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, set='train', list_file='', scale_aug=False, rot_aug=False, test_mode=True, num_views=12,
                 shuffle=False, data_type='', abstract=0.5, view_type=''):

        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.name_list = load_string_list(list_file.format(set))
        self.file_paths = []
        self.labels = []
        self.sketch_end = '_sketch_{}_'.format(abstract)

        if abstract > 1.0 and data_type=='sketch':
            data_type = 'human'

        stride = int(12 / self.num_views)  # 12 6 4 3 2 1
        all_files = []
        view_dir = os.path.join(VIEW_DIR, data_type+'_'+view_type)
        if data_type == 'shape':
            for line in self.name_list:
                model_name, class_id = line.split(' ')
                shape_paths = [(view_dir+ '/' + model_name + '_opt_{}.png').format(i) for i in range(12)]
                all_files.extend(shape_paths)  # Edge
                self.labels.append(int(class_id))
        elif data_type == 'sketch':
            for line in self.name_list:
                model_name, class_id = line.split(' ')
                sketch_paths = [(view_dir+ '/' + model_name + self.sketch_end + '{}.png').format(i) for i in range(12)]
                all_files.extend(sketch_paths)
                self.labels.append(int(class_id))
        elif data_type == 'human':
            human_list = '/yrfs1/intern/lingluo3/data/list/ModelNet10/equal/human_test.txt'
            name_list = [line.rstrip() for line in open(human_list)]
            for line in name_list:
                model_name, class_id = line.split(' ')
                sketch_paths = [(view_dir+ '/' + model_name + '_{}.png').format(i) for i in range(12)]
                all_files.extend(sketch_paths)
                self.labels.append(int(class_id))

        ## Select subset for different number of views
        all_files = all_files[::stride]
        self.file_paths = all_files

        print('The size of %s data is %d' % (set, len(self.labels)))
        if shuffle == True:
            # permute
            rand_idx = np.random.permutation(int(len(self.file_paths) / num_views))
            filepaths_new = []
            labels_new = []
            anchors_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.file_paths[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
                labels_new.append(self.labels[i])
                anchors_new.append(self.anchors[i])
            self.file_paths = filepaths_new
            self.labels = labels_new
            self.anchors = anchors_new

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
        return int(len(self.file_paths) / self.num_views)

    def __getitem__(self, idx):
        class_id = self.labels[idx]
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.file_paths[idx * self.num_views + i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        # return (class_id, torch.stack(imgs), self.file_paths[idx * self.num_views:(idx + 1) * self.num_views])
        return (torch.stack(imgs), class_id)

if __name__ == "__main__":
    from config import DATASETS
    dataset = 'ModelNet10'
    list_file = DATASETS[dataset]['list_file']
    # train_dataset = MultiviewImgDataset(set='train', list_file=list_file, scale_aug=False, rot_aug=False, num_views=12,
    #                                     shuffle=False)
    # print(train_dataset.__len__(), len(train_dataset.file_paths))
    train_dataset = MultiviewImgDataset(set='valid', list_file=list_file, data_type='sketch', view_type='depth', abstract=2.0)
    # print(train_dataset.__len__(), len(train_dataset.file_paths))

    # train_dataset = MultiviewImgDataset(set='test', list_file=list_file, scale_aug=False, rot_aug=False, num_views=12,
    #                                     shuffle=False)
    # print(train_dataset.__len__(), len(train_dataset.file_paths))


    # train_dataset = MultiviewImgDataset(set='valid', list_file=list_file, scale_aug=False, rot_aug=False, num_views=12,
    #                                     shuffle=False, dataset=dataset)

    DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

    # for labels, images in DataLoader:
    for _, data in enumerate(DataLoader, 0):
        print(data[1].shape)
        # print(images.shape)
