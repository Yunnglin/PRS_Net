import os
import random

import nrrd
import torch
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms
import yaml


def rand_split_data(train_proportion=0.8):
    conf_path = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(conf_path, 'config\\config.yml')
    f = open(yamlPath, 'r', encoding='utf-8')
    cont = f.read()
    x = yaml.load(cont)
    data_path = x['data']['data_root']

    data_list = list(range(1, x['data']['size']+1))
    train_list = random.sample(data_list, int(len(data_list)*train_proportion))
    train_list.sort()
    test_list = [n for n in data_list if n not in train_list]

    with open(os.path.join(data_path, 'train.csv'), 'w') as train:
        for i in train_list:
            train.write(str(i)+'\n')

    with open(os.path.join(data_path, 'test.csv'), 'w') as test:
        for i in test_list:
            test.write(str(i)+'\n')


class PRSDataset(Dataset.Dataset):
    def __init__(self, data_path, isTrain=True):
        self.data_path = data_path
        self.data_list = []
        self.size = 0
        self.transform = transforms.ToTensor()

        if isTrain:
            self.data_file = 'train.csv'
        else:
            self.data_file = 'test.csv'

        with open(os.path.join(self.data_path, self.data_file)) as file:
            for i in file:
                self.data_list.append(i.strip('\n'))
                self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # read nrrd data
        nrrd_path = os.path.join(
            self.data_path, self.data_list[index], 'model.nrrd')
        nrrd_data, head = nrrd.read(nrrd_path)
        voxel = self.transform(nrrd_data)
        voxel = voxel.view(1, 32, 32, 32)

        # read point cloud data
        pcd_path = os.path.join(
            self.data_path, self.data_list[index], 'model_uniform.pcd')
        with open(pcd_path, 'r') as pcd_data:
            points = []
            for i in pcd_data:
                line = i.strip('\n')
                point = line.split(' ')
                points.append([float(point[0]),
                               float(point[1]),
                               float(point[2])])
        points = torch.tensor(points)

        # read nearest point of each voxel
        nstpoint = torch.zeros(1,32,32,32)
        npv_path = os.path.join(
            self.data_path, self.data_list[index], 'model.npv')
        with open(npv_path,'r') as npv_data:
            line = npv_data.readline()
            indices = line.strip('\n').split(' ')
            count=0
            for i in range(32):
                for j in range(32):
                    for k in range(32):
                        nstpoint[0, i, j, k] = int(indices[count])
                        count += 1
        
        print('Read {}th sample is OK'.format(index))

        sample = {
            'voxel':voxel,
            'points':points,
            'nearest':nstpoint
        }
        return sample
        
    
