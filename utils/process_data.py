import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import nrrd
import random
import os

data_root = 'D:\\My\\MyCode\\PRSPaper\\PRSNet\\data\\'
start_file = 42
file_count = 151  # 151
view = False

for index in range(start_file, file_count+1):
    # read nrrd voxel data
    nrrd_path = os.path.join(data_root, str(index), 'model.nrrd')
    nrrd_data, nrrd_head = nrrd.read(nrrd_path)
    # print(nrrd_head)
    transform = transforms.ToTensor()
    voxel = transform(nrrd_data)
    voxel = voxel.view(1, 32, 32, 32)
    # print(voxel)

    # read pcd points data
    pcd_path = os.path.join(data_root, str(index), 'model.pcd')
    with open(pcd_path, mode='r') as pcd_data:
        # skip first 9 lines
        for i in range(9):
            pcd_data.readline()
        num_line = pcd_data.readline()
        num_points = num_line.split(' ')[1]
        pcd_data.readline()  # skip one more line
        points = []
        for i in range(int(num_points)):
            line = pcd_data.readline()
            point = line.split(' ')
            points.append([float(point[0]),
                           float(point[1]),
                           float(point[2])])

    # add up to 1000
    if int(num_points) < 1000:
        if (1000 - int(num_points)) > int(num_points):
            print('skip {}th file'.format(index)) 
            continue
            
        addition = random.sample(points, 1000 - int(num_points))
        points.extend(addition)
    # uniform sample points
    points = torch.tensor(points)
    max, _ = torch.max(points, dim=0)
    min, _ = torch.min(points, dim=0)
    max_dis = torch.max(max - min)
    points = points - min
    points = points / max_dis * 32

    # view voxel
    if view:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', s=10)
        ax.view_init(45, 45)
        ax.voxels(voxel[0], color='w', edgecolor="b")
        figPath = os.path.join(data_root, str(index), "model.png")
        fig.savefig(figPath)

    # update point cloud data
    update_pcd_path = os.path.join(data_root, str(index), 'model_uniform.pcd')
    with open(update_pcd_path, mode='w') as pcd_data:
        for i in range(len(points)):
            pcd_data.write('{} {} {}\n'.format(float(points[i][0]),
                                               float(points[i][1]),
                                               float(points[i][2])))

    # precompute the closest point on the surface
    nearest_point_of_voxel = torch.zeros(1, 32, 32, 32)
    for i in range(32):
        for j in range(32):
            for k in range(32):
                voxel_pos = torch.tensor([i+0.5, j+0.5, k+0.5])
                distance = torch.norm(points-voxel_pos, dim=1, keepdim=True)
                _, point_index = torch.min(distance, dim=0)
                nearest_point_of_voxel[0, i, j, k] = point_index

    # save closest point of a regular grid
    nvp_path = os.path.join(data_root, str(index), 'model.npv')
    with open(nvp_path, 'w') as nvp_data:
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    nvp_data.write('{} '.format(
                        int(nearest_point_of_voxel[0, i, j, k])))

    print('{}th sample is OK'.format(index))
