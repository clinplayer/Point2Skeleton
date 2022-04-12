import numpy as np
import warnings
import os
from torch.utils.data import Dataset
import torch
import math
import FileRW as rw


class PCDataset(Dataset):
    def __init__(self, data_list, data_folder, point_num):
        self.data_id = data_list
        self.data_folder = data_folder
        self.point_num = point_num

    def __getitem__(self, index):
        data_pc = rw.load_ply_points(self.data_folder + self.data_id[index] + '.ply', expected_point=self.point_num)
        return index, data_pc

    def __len__(self):
        return len(self.data_id)
