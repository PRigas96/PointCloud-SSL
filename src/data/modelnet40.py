import torch
import pytorch3d
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pytorch3d.datasets import ModelNet
from pytorch3d.ops import sample_points_from_meshes
from path import Path
from pytorch3d.io import load_obj, IO
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

# Disable telemetry logging
os.environ["IOPATH_LOGGING"] = "off"

class ModelNet40Dataset(Dataset):
    def __init__(
        self,
        data_dir,
        classes_file,
        num_points=1024,
        split="train",
        transform=None,
        resampling=False,
        return_normals=False,
        normalize=True,
    ):
        self.data_dir = data_dir
        self.num_points = num_points
        self.split = split  # 'train' or 'test'
        if self.split not in ["train", "test"]:
            raise ValueError("Invalid split. Must be 'train' or 'test'.")
        self.transform = transform
        self.resampling = resampling
        self.return_normals = return_normals
        self.normalize = normalize

        # Read classes from the text file
        with open(classes_file, "r") as f:
            self.classes = f.read().splitlines()

        self.data = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name, self.split)
            if not os.path.isdir(class_dir):
                print(f"Directory {class_dir} does not exist.")
                continue
            for mesh_file in os.listdir(class_dir):
                if mesh_file.endswith(".off"):
                    mesh_path = os.path.join(class_dir, mesh_file)
                    self.data.append(mesh_path)
                    self.labels.append(label)
            if len(self.data) == 0:
                print(f"No valid .off files found in {class_dir}.")
            else:
                print(f"Loaded {len(self.data)} meshes from {class_dir}.")

    def __len__(self):
        return len(self.data)

    def _normalize_point_cloud(self, point_cloud):
        mean = point_cloud.mean(dim=0)
        std = point_cloud.std(dim=0)
        point_cloud = (point_cloud - mean) / std
        return point_cloud

    def __getitem__(self, idx):
        mesh_path = self.data[idx]
        label = self.labels[idx]

        # Load the mesh
        mesh = IO().load_mesh(mesh_path)

        # Sample points from the mesh
        point_cloud, normals = sample_points_from_meshes(
            mesh, num_samples=self.num_points, return_normals=True
        )
        point_cloud = point_cloud.squeeze(0)
        point_cloud = self.transform(point_cloud) if self.transform else point_cloud
        point_cloud = (
            self._normalize_point_cloud(point_cloud) if self.normalize else point_cloud
        )

        return_tuple = (point_cloud,)
        if self.resampling:
            target_point_cloud = sample_points_from_meshes(
                mesh, num_samples=self.num_points
            ).squeeze(0)
            target_point_cloud = (
                self.transform(target_point_cloud)
                if self.transform
                else target_point_cloud
            )
            target_point_cloud = (
                self._normalize_point_cloud(target_point_cloud)
                if self.normalize
                else target_point_cloud
            )
            return_tuple += (target_point_cloud,)
        if self.return_normals:
            normals = normals.squeeze(0)
            return_tuple += (normals,)
        return_tuple += (label,)

        return return_tuple

