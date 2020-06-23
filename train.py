from torch.utils.data import DataLoader
import torch
from PRSDataset import PRSDataset
import PRSNet as PN
import os

batch_size = 4
lr = 0.01
epochs = 4
w_r = 100


def train():
    train_set = PRSDataset('.\\data', True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    PRS_Net = PN.PRSNet()
    LossSymmetryDistance = PN.LossSymmetryDistance()
    LossRegularization = PN.LossRegularization()

    optimizer = torch.optim.Adam(PRS_Net.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, sample in enumerate(train_loader):
            voxel = sample['voxel']
            optimizer.zero_grad()
            outputs = PRS_Net(voxel)
            l_sd = LossSymmetryDistance(outputs, sample)
            l_r = LossRegularization(outputs)

            l_sd_ = torch.sum(l_sd) / batch_size
            l_r_ = torch.sum(l_r) / batch_size
            loss = l_sd_ + w_r*l_r_
            loss.backward()
            optimizer.step()
            status = "{}th epoch, {}th mini-batch, \
lsd is {}, lreg is {}, outputs are {}\n".format(epoch, i, l_sd_, l_r_, outputs)
            print(status)
            
            with open(os.path.join('.\\log','train_log.txt'),'a') as file:
                file.write(status)
                
        torch.save(PRS_Net, os.path.join('.\\model','PRS_Net{}.pkl'.format(str(epoch))))

if __name__=='__main__':
    train()