# this module is used to load ModelNet10 dataset
import torch


class ModelNet10:
    def __init__(self, root: str, transforms=None, train: bool = True):
        self.root = root
        self.transforms = transforms
        self.train = train
        self.data = []
        self.labels = []
        self.classes = {}
        self.class_map = {}
        self.load_data()
        self.get_classes()
        self.get_class_map()
        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)

    def load_data(self):
        if self.train:
            with open(f"{self.root}/modelnet10_train.txt", "r") as f:
                lines = f.readlines()
        else:
            with open(f"{self.root}/modelnet10_test.txt", "r") as f:
                lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            self.data.append(torch.load(f"{self.root}/{line[0]}"))
            self.labels.append(int(line[1]))

    def get_classes(self):
        classes = []
        for label in self.labels:
            if label not in classes:
                classes.append(label)
        self.classes = classes

    def get_class_map(self):
        for i, label in enumerate(self.classes):
            self.class_map[label] = i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample, self.class_map[label]

    def get_class_name(self, label):
        for key, value in self.class_map.items():
            if value == label:
                return key
        return None

    def get_class_map(self):
        return self.class_map

    def get_classes(self):
        return self.classes


# Path: src/data/transforms.py
if __name__ == "__main__":
    from transforms import PointSampler
    import matplotlib.pyplot as plt
    import numpy as np

    path = "data/ModelNet10/"
    data = ModelNet10(root=path, train=True)
