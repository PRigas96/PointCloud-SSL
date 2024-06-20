import torch
import h5py
from path import Path


class ModelNet40(torch.utils.data.Dataset):
    """
    ModelNet40 dataset. contains 40 classes of CAD models, 1000 models per class.
    Each model is represented as a point cloud with 2048 points.
    """

    def __init__(
        self,
        root: str,
        transforms=None,
        train: bool = True,
    ):
        super(ModelNet40, self).__init__()
        self.root = Path(root)
        self.train_data = []
        self.train_idx = 5
        self.train_labels = []
        self.test_data = []
        self.test_idx = 2
        self.test_labels = []
        self.classes = {}
        self.shape_names = self.root / "shape_names.txt"
        self.name_files = [self.root / "train_files.txt", self.root / "test_files.txt"]
        self._load_data()
        self._get_classes()
        self.data = self.train_data if train else self.test_data
        self.labels = [self.train_labels, self.test_labels]
        self.idx = [self.train_idx, self.test_idx]

    def __getitem__(self, index, train=True):
        # TODO: check if train should be taken from __init__ or from the function
        if torch.is_tensor(index):
            index = index.tolist()
        if train:
            data = self.train_data[index]
            label = self.train_labels[index]
        else:
            data = self.test_data[index]
            label = self.test_labels[index]
        return {"data": data, "label": label}

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        names = []
        for i, name_file in enumerate(self.name_files):
            with open(name_file, "r") as f:
                for _, line in enumerate(f):
                    names.append(line.strip())
        for ind in range(names.__len__()):
            filename = "./src/" + names[ind]
            f = h5py.File(filename, "r")
            if ind < self.train_idx:
                self.train_data.append(torch.tensor(f["data"][:]))
                self.train_labels.append(torch.tensor(f["label"][:]))
            else:
                self.test_data.append(torch.tensor(f["data"][:]))
                self.test_labels.append(torch.tensor(f["label"][:]))
            f.close()
            # concatenate all the data in one tensor
        self.train_data = torch.cat(self.train_data, dim=0)
        self.test_data = torch.cat(self.test_data, dim=0)
        self.train_labels = torch.cat(self.train_labels, dim=0)
        self.test_labels = torch.cat(self.test_labels, dim=0)

    def _get_classes(self):
        # get self.shape_names open it and then make a dictionary of classes
        with open(self.shape_names, "r") as f:
            for i, line in enumerate(f):
                self.classes[line.strip()] = i

    def __str__(self):
        return (
            "ModelNet40 dataset\n"
            "===================\n"
            f"Classes: {len(self.classes)} \n"
            f"{self.classes}\n"
        )


if __name__ == "__main__":
    modelnet40 = ModelNet40(root="./src/data/modelnet40_hdf5_2048")
    # get dataloader
    BATCH_SIZE = 32
    dataloader = torch.utils.data.DataLoader(
        modelnet40, batch_size=BATCH_SIZE, shuffle=False
    )
    # iterate over the dataset
    for i_batch, batched_data in enumerate(dataloader):
        data = batched_data["data"]
        label = batched_data["label"]
        print(f"Batch {i_batch} data size: {data.size()}")
        print(f"Batch {i_batch} label size: {label.size()}")
        break
