import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import plotly.graph_objects as go
import plotly.express as px

from path import Path
import utils
import Dataset

import model.model

def read_off(file):
    if "OFF" != file.readline().strip():
        raise ("Not a valid OFF header")
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(" ")])
    verts = [
        [float(s) for s in file.readline().strip().split(" ")]
        for i_vert in range(n_verts)
    ]
    faces = [
        [int(s) for s in file.readline().strip().split(" ")][1:]
        for i_face in range(n_faces)
    ]
    return verts, faces

def pcshow(xs,ys,zs):
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = utils.visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()


def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)


def train(model, train_loader, val_loader=None, epochs=15, save=True):
    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        if save:
            torch.save(model.state_dict(), "save_" + str(epoch) + ".pth")

def main():
    path = Path(os.getcwd() + "/data/ModelNet10")
    print("your data path is: {}".format(path))

    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
    classes = {folder: i for i, folder in enumerate(folders)}
    print(classes)

    with open(path / "bed/train/bed_0001.off", "r") as f:
        verts, faces = read_off(f)

    i, j, k = np.array(faces).T
    x, y, z = np.array(verts).T

    print(len(x))

    #utils.visualize_rotate(
        #[go.Mesh3d(x=x, y=y, z=z, color="lightpink", opacity=0.50, i=i, j=j, k=k)]
    #).show()

    #utils.visualize_rotate([go.Scatter3d(x=x, y=y, z=z,
                                   #mode='markers')]).show()

    #pcshow(x,y,z)

    pointcloud = utils.PointSampler(3000)((verts, faces))
    #pcshow(*pointcloud.T)
    
    norm_pointcloud = utils.Normalize()(pointcloud)
    #pcshow(*norm_pointcloud.T)

    rot_pointcloud = utils.RandRotation_z()(norm_pointcloud)
    noisy_rot_pointcloud = utils.RandomNoise()(rot_pointcloud)

    #pcshow(*noisy_rot_pointcloud.T)

    #print(utils.ToTensor()(noisy_rot_pointcloud))

    train_transforms = transforms.Compose([
                    utils.PointSampler(1024),
                    utils.Normalize(),
                    utils.RandRotation_z(),
                    utils.RandomNoise(),
                    utils.ToTensor()
                    ])

    train_ds = Dataset.PointCloudData(path, transform=train_transforms)
    valid_ds = Dataset.PointCloudData(path, valid=True, folder='test', transform=train_transforms)

    train_ds = Dataset.PointCloudData(path, transform=train_transforms)
    valid_ds = Dataset.PointCloudData(path, valid=True, folder='test', transform=train_transforms)

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print(inv_classes)

    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())
    print('Class: ', inv_classes[train_ds[0]['category']])

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    pointnet = model.model.PointNet()
    pointnet.to(device)
    print(pointnet)

    optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)

    train(pointnet,train_loader,valid_loader,save=False)

if __name__ == "__main__":
    main()
