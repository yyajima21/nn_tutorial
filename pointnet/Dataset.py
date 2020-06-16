from torch.utils.data import Dataset, DataLoader
import utils
import os
from path import Path

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

class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=utils.default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else utils.default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud, 
                'category': self.classes[category]}