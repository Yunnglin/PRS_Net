from torch import nn
import torch
import time
from utils import quaternion


class PRSNet(nn.Module):
    def __init__(self):
        super(PRSNet, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        # convolution layers
        self.ConvLayer1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.ConvLayer2 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.ConvLayer3 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.ConvLayer4 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.ConvLayer5 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2), self.LeakyReLU)

        self.ConvLayers = nn.Sequential(
            self.ConvLayer1,
            self.ConvLayer2,
            self.ConvLayer3,
            self.ConvLayer4,
            self.ConvLayer5,
        )

        # Fully connected layers of symmetry planes and rotation quternion
        self.FCLayers = nn.Sequential(
            nn.Linear(64, 32),
            self.LeakyReLU,
            nn.Linear(32, 16),
            self.LeakyReLU,
            nn.Linear(16, 4),
            self.LeakyReLU
        )
        
        self.FCLayerSP11 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerSP21 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerSP31 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerSP12 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerSP22 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerSP32 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerSP13 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)
        self.FCLayerSP23 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)
        self.FCLayerSP33 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)

        self.FCLayerRQ11 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerRQ21 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerRQ31 = nn.Sequential(nn.Linear(64, 32), self.LeakyReLU)
        self.FCLayerRQ12 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerRQ22 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerRQ32 = nn.Sequential(nn.Linear(32, 16), self.LeakyReLU)
        self.FCLayerRQ13 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)
        self.FCLayerRQ23 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)
        self.FCLayerRQ33 = nn.Sequential(nn.Linear(16, 4), self.LeakyReLU)

    def forward(self, voxel):
        print('voxel1', voxel.shape)
        self.outputs = torch.zeros(voxel.shape[0], 6, 4)  # (4,6,4) batchsize
        # convolution Layers & pool Layers extract the global features of the shape
        voxel = self.ConvLayers(voxel)
        voxel = voxel.view(voxel.shape[0], 64)
        print('voxel1', voxel.shape)

        # three plane and three rotation axes, each has 4 parameters
        a = self.FCLayerSP11(voxel)
        a = self.FCLayerSP12(a)
        a = self.FCLayerSP13(a)
        self.assign2Outputs(self.unitlize(a), 0)
        # print(a)

        a = self.FCLayerSP21(voxel)
        a = self.FCLayerSP22(a)
        a = self.FCLayerSP23(a)
        self.assign2Outputs(self.unitlize(a), 1)

        a = self.FCLayerSP31(voxel)
        a = self.FCLayerSP32(a)
        a = self.FCLayerSP33(a)
        self.assign2Outputs(self.unitlize(a), 2)

        a = self.FCLayerRQ11(voxel)
        a = self.FCLayerRQ12(a)
        a = self.FCLayerRQ13(a)
        self.assign2Outputs(self.unitlize(a), 3)

        a = self.FCLayerRQ21(voxel)
        a = self.FCLayerRQ22(a)
        a = self.FCLayerRQ23(a)
        self.assign2Outputs(self.unitlize(a), 4)

        a = self.FCLayerRQ31(voxel)
        a = self.FCLayerRQ32(a)
        a = self.FCLayerRQ33(a)
        self.assign2Outputs(self.unitlize(a), 5)

        return self.outputs

    def unitlize(self, n):
        return n/torch.norm(n, p=2, dim=1)

    def assign2Outputs(self, x: torch.tensor, index: int):
        for i in range(self.outputs.shape[0]):
            self.outputs[i][index] = x[i]
        return

    def __call__(self, voxel):
        # batchsize*1*32*32*32
        return self.forward(voxel)


# to promote planar symmetry
# chapter 4.1
class LossSymmetryDistance(object):
    def __call__(self, outputs: torch.tensor, sample):
        self.loss = torch.zeros(outputs.shape[0], 6)
        for i in range(outputs.shape[0]):
            self.voxel = sample['voxel'][i]
            self.points = sample['points'][i]
            self.nstpoint = sample['nearest'][i]

            for j in range(0, 3):
                self.processed_points = self.symmTransform(
                    outputs[i][j])  # plane
                self.loss[i][j] = self.sumAllDistance()

            for j in range(3, 6):
                self.processed_points = self.rotateTransform(
                    outputs[i][j])  # axis
                self.loss[i][j] = self.sumAllDistance()

        return self.loss

    def sumAllDistance(self):
        sum_distance = 0
        for i in range(self.points.shape[0]):
            x = int(self.processed_points[i][0])
            y = int(self.processed_points[i][1])
            z = int(self.processed_points[i][2])
            x = 31 if x > 31 else x
            y = 31 if y > 31 else y
            z = 31 if z > 31 else z
            x = 0 if x < 0 else x
            y = 0 if y < 0 else y
            z = 0 if z < 0 else z
            index = int(self.nstpoint[0, x, y, z])
            d = torch.norm(self.points[index] - self.processed_points[i])
            sum_distance += d
        return sum_distance

    def symmTransform(self, plane_parameters: torch.tensor):
        outPoints = torch.zeros_like(self.points)  # n*3
        for i in range(self.points.shape[0]):
            outPoints[i] = self.points[i] - 2 * plane_parameters[0:3] * (
                torch.dot(self.points[i], plane_parameters[0:3]) +
                plane_parameters[3]) / torch.dot(plane_parameters[0:3],
                                                 plane_parameters[0:3])
        return outPoints

    def rotateTransform(self, q: torch.tensor):
        outPoints = torch.zeros_like(self.points)
        for i in range(self.points.shape[0]):
            outPoints[i] = quaternion.rotate(q, self.points[i])
        return outPoints


# to avoid producing duplicated symmetry planes
# chapter 4.2
class LossRegularization(object):
    def __call__(self, outputs: torch.tensor):
        self.loss = torch.zeros(outputs.shape[0])  # tensor([0., 0., 0., 0.])
        for i in range(outputs.shape[0]):
            M1 = outputs[i][0:3, 0:3]
            M2 = outputs[i][3:6, 1:4]
            # print('M1, M2', M1, M2)
            M1 = M1 / torch.norm(M1, dim=1).view(-1, 1)
            M2 = M2 / torch.norm(M2, dim=1).view(-1, 1)
            diagone = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            A = torch.mm(M1, M1.t()) - diagone
            B = torch.mm(M2, M2.t()) - diagone
            # print('A', A)
            # print('B', B)
            self.loss[i] = torch.sum(A**2) + torch.sum(B**2)

            # print('loss', i, self.loss[i])
        # print('loss', self.loss)
        return self.loss


# to remove duplicated outputs: if its dihedral angle is less than π/6
# symmetry planes/rotation axes lead to high symmetry distance loss greater than 4 × 10−4
# figure 3 shows the examples
# chapter 5
class ValidateOutputs(object):
    def __call__(self, outputs: torch.tensor, lsd: torch.tensor, ml: float,
                 mc: float):
        self.isRemoved = [False, False, False, False, False, False]
        for i in range(6):
            if lsd[i] > ml:
                self.isRemoved[i] = True
        for i in range(2):
            if self.isRemoved[i] is True:
                continue
            for j in range(i + 1, 3):
                if self.isRemoved[j] is True:
                    continue
                if self.cosDihedralAngle(outputs[i][0:3],
                                         outputs[j][0:3]) > mc:
                    if lsd[i] > lsd[j]:
                        self.isRemoved[i] = True
                    else:
                        self.isRemoved[j] = True

        for i in range(3, 5, 1):
            if self.isRemoved[i] is True:
                continue
            for j in range(i + 1, 6):
                if self.isRemoved[j] is True:
                    continue
                if self.cosDihedralAngle(outputs[i][1:4],
                                         outputs[j][1:4]) > mc:
                    if lsd[i] > lsd[j]:
                        self.isRemoved[i] = True
                    else:
                        self.isRemoved[j] = True

        for i in range(6):
            if self.isRemoved[i] is True:
                outputs[i] = torch.zeros(4)
        return outputs

    def cosDihedralAngle(self, normal1: torch.tensor, normal2: torch.tensor):
        return torch.abs(
            torch.dot(normal1, normal2) /
            (torch.norm(normal1) * torch.norm(normal2)))
