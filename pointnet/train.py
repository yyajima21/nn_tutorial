"Reference: https://github.com/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb"

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

    utils.visualize_rotate(
        [go.Mesh3d(x=x, y=y, z=z, color="lightpink", opacity=0.50, i=i, j=j, k=k)]
    ).show()

    #visualize_rotate([go.Scatter3d(x=x, y=y, z=z,
                                   #mode='markers')]).show()


if __name__ == "__main__":
    main()
