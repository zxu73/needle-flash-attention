import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        images, labels = [], []
        for name in names:
            # each batch file is a pickled dict with byte-string keys:
            # b"data" is (10000, 3072) uint8, rows laid out as R(1024) G(1024) B(1024)
            with open(os.path.join(base_folder, name), "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            images.append(batch[b"data"])
            labels.extend(batch[b"labels"])
        # channels-first (N, 3, 32, 32), scaled to [0, 1]
        self.X = (
            np.concatenate(images).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        )
        self.y = np.array(labels, dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X, y = self.X[index], self.y[index]
        if self.transforms is not None:
            # the transforms work on a single H x W x C image
            def tform(img):
                return self.apply_transforms(img.transpose(1, 2, 0)).transpose(2, 0, 1)

            if X.ndim == 3:
                X = tform(X)
            else:
                X = np.stack([tform(img) for img in X])
        return X, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
