import os
from glob import glob
import pickle
import numpy as np
import pandas as pd
import torch
from deeplearning.datasets import Dataset


class AudioDataset(Dataset):


    def __init__(self, data_path, standardization_path, **kwargs):

        super().__init__(**kwargs)

        self.means, self.stds = pickle.load(open(standardization_path, 'rb'))
        self.means, self.stds = self.means[:, None], self.stds[:, None]

        genre_folders = glob(os.path.join(data_path, '**'))
        genres = [x.split('/')[-1] for x in genre_folders]

        self.genres_to_int = {genre : i for i, genre in enumerate(genres)}

        print(self.genres_to_int)

        feature_files = glob(os.path.join(data_path, '**/*.npy'))

        max = 0


        for i, feature_file in enumerate(feature_files):


            features = np.load(feature_file)
            length = features.shape[1]

            if length > max:
                max = length

        self.max_length = max
        self.feature_files = feature_files


    def __getitem__(self, index):

        feature_file = self.feature_files[index]

        features = np.load(feature_file)

        features = (features - self.means) / self.stds

        length = features.shape[1]
        target = self.genres_to_int[feature_file.split('/')[-2]]

        features = np.pad(features, ((0,0),(0, self.max_length - length)), mode='constant')

        return (torch.FloatTensor(features.T), length),  target

    def __len__(self):
        return len(self.feature_files)

    @staticmethod
    def args(parser):

        parser.add_argument('--data_path')
        parser.add_argument('--standardization_path')


        return super(AudioDataset, AudioDataset).args(parser)
