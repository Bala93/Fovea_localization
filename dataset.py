import torch
import numpy as np
import cv2
from PIL import Image, ImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import pandas as pd
import os


class DatasetImageCoord(Dataset):
    def __init__(self, data_path, img_csv_path, pos_csv_path):

        df = pd.read_csv(img_csv_path)
        fnames = df["data"].to_list()

        df = pd.read_csv(pos_csv_path)

        xlist = df["x"].to_list()
        ylist = df["y"].to_list()

        self.file_names = []
        self.positions = []
        self.data_path = data_path

        for ind in fnames:
            self.file_names.append("{0:04}".format(ind))
            self.positions.append(
                [xlist[ind - 1], ylist[ind - 1]]
            )  # filename starts from 1

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        position = self.positions[idx]

        img_path = os.path.join(
            self.data_path, img_file_name, "{}.jpg".format(img_file_name)
        )
        image, dim = load_image(img_path)

        #print (position, dim)

        # normalize the coord
        x = position[0] / dim[1]
        y = position[1] / dim[0]

        pos = torch.from_numpy(np.array([x,y]))
        dim = torch.from_numpy(np.array(dim))

        #print (position)

        return image, pos, dim, img_file_name


def load_image(path):

    img = Image.open(path)
    dim = img.size
    data_transforms = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = data_transforms(img)

    return img, dim
