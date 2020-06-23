import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import PRSNet as PN
from PRSDataset import PRSDataset

cos_dihedral_angle = 0.866025
symmetry_distance_loss = 2500


def test():
    test_data = PRSDataset('.\\data', False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    PRS_Net = torch.load(os.path.join('.\\model', 'PRS_Net3.pkl'))
    LossSymmetryDistance = PN.LossSymmetryDistance()
    LossRegularization = PN.LossRegularization()
    ValidateOutputs = PN.ValidateOutputs()

    for i, sample in enumerate(test_loader):
        voxel = sample['voxel']
        outputs = PRS_Net(voxel)
        l_sd = LossSymmetryDistance(outputs, sample)
        l_r = LossRegularization(outputs)
        outputs = outputs.view(6, 4)
        l_sd = l_sd.view(6)
        l_r = l_r.view(1)
        outputs = ValidateOutputs(
            outputs, l_sd, symmetry_distance_loss, cos_dihedral_angle)

        status = "{}th sample, lsd for each symmetry plane and rotation axis \
 is {}, lreg is {}, validated outputs is {}\n".format(i, l_sd, l_r, outputs)
        print(status)

        # visualize
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.view_init(45, 45)
        ax.set_xlim(-4, 36)
        ax.set_ylim(-4, 36)
        ax.set_zlim(-4, 36)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        X = np.arange(-4, 36, 1)
        Y = np.arange(-4, 36, 1)
        X, Y = np.meshgrid(X, Y)
        for j in range(0, 3, 1):
            if torch.sum(outputs[j]) > 0.01:
                a = float(outputs[j][0])
                b = float(outputs[j][1])
                c = float(outputs[j][2])
                d = float(outputs[j][3])
                c = 0.01 if c < 0.01 else c
                Z = (-1 * d - a * X - b * Y) / c
                ax.plot_surface(X,
                                Y,
                                Z,
                                rstride=1,
                                cstride=1,
                                cmap='rainbow')
        ax.voxels(voxel[0, 0], color='w', edgecolor="b")
        figPath = os.path.join('.\\results', str(i) + '.png')
        fig.savefig(figPath)


if __name__=='__main__':
    test()
