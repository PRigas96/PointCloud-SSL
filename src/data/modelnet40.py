import torch
import pytorch3d
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pytorch3d.datasets import ModelNet
from pytorch3d.ops import sample_points_from_meshes
from path import Path
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

class ModelNet40(torch.utils.data.Dataset):
    """
    ModelNet40 dataset. contains 40 classes of CAD models, 1000 models per class.
    Each model is represented as a point cloud with 2048 points.
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        transforms: None = None,
    ):
        super(ModelNet40, self).__init__()
        self.root = Path(root)
        self.data = None
        self.labels = None
        self.verts, self.faces = self._read_off()
        self.data, self.labels = self._get_data()
        
    
    def _read_off(self):
        with open(self.root, "r") as file:
            if "OFF" != file.readline().strip():
                raise ("Not a valid OFF header")
            n_verts, n_faces, _ = list(map(int, file.readline().strip().split(" ")))
            verts = [
                [float(x) for x in file.readline().strip().split(" ")] for _ in range(n_verts)
            ]
            faces = [
                [int(x) for x in file.readline().strip().split(" ")][1:] for _ in range(n_faces)
            ]
            return verts, faces
        
        

