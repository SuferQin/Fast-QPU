from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name,'r')
    data = f["data"][:]
    label = f["label"][:]
    return data, label

def prepare_data(root,download=True):
    mn40_data_dir = os.path.join(root, "modelnet40_ply_hdf5_2048")
    url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
    if download and not os.path.exists(mn40_data_dir):
        zipfile = os.path.join(root, os.path.basename(url))
        subprocess.check_call(
            shlex.split("curl {} -o {}".format(url, zipfile))
        )
        subprocess.check_call(
            shlex.split("unzip {} -d {}".format(zipfile, root))
        )
        subprocess.check_call(shlex.split("rm {}".format(zipfile)))

    mn10_data_dir = os.path.join(root.replace('40','10'), "modelnet10_ply_hdf5_2048")
    if not os.path.exists(mn10_data_dir):
        label_name_10 = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        all_label_name = open(os.path.join(mn40_data_dir,'shape_names.txt')).readlines()
        Label2Id = {it[:-1]:idx for idx,it in enumerate(all_label_name)}
        label10 = [Label2Id[name] for name in label_name_10]
        Lab40_to_Lab10 = {it:idx for idx,it in enumerate(label10)}
        os.makedirs(mn10_data_dir)
        for file_name in ["train_files.txt","test_files.txt"]:
            files = _get_data_files(os.path.join(mn40_data_dir, file_name))
            point_list, label_list = [], []
            for f in files:
                points, labels = _load_data_file(os.path.join(root, f))
                sample_idx = np.array([it[0] in label10 for it in labels])
                points = points[sample_idx]
                labels = np.array([[Lab40_to_Lab10[it[0]]] for it in labels[sample_idx]]).astype(np.uint8)
                point_list.append(points)
                label_list.append(labels)
            points = np.concatenate(point_list, 0)
            labels = np.concatenate(label_list, 0)
            out_name = file_name.split('_')[0]+'.h5'
            hf = h5py.File(os.path.join(mn10_data_dir,out_name),'w')
            hf.create_dataset('data',data=points)
            hf.create_dataset('label',data=labels)
            hf.close()
    return mn10_data_dir


class ModelNet10Cls(data.Dataset):
    def __init__( self, num_points, root, transforms=None, train=True, download=True):
        super().__init__()

        self.transforms = transforms
        self.data_dir = prepare_data(root,download)
        
        self.train = train
        file_name = 'train.h5' if self.train else 'test.h5'

        self.points, self.labels = _load_data_file(os.path.join(self.data_dir,file_name))
        self.set_num_points(num_points)

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])
        np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

        if self.transforms is not None:
            current_points = self.transforms(current_points)

        return current_points, label

    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = min(self.points.shape[1], pts)

    def randomize(self):
        pass

if __name__ == "__main__":
    pass
