from deeplearning.datasets import Dataset
from glob import glob
import os
import pandas as pd
import numpy as np
import torch
class AudioDataset(Dataset):


    def __init__(self, data_path, **kwargs):

        super().__init__(**kwargs)

        feature_files = glob(os.path.join(data_path, '**/*.npy'))


        data = pd.DataFrame({'file_path' : feature_files})
        data['class_name'] = data['file_path'].apply(lambda x : x.split('/')[-2])
        data['target'] = pd.factorize(data['class_name'])[0]
        data['mfcc'] = data['file_path'].apply(lambda x : np.load(x))
        data['length'] = data['mfcc'].apply(lambda x : x.shape[1])

        longest = data['length'].max()

        data['mfcc'] = data['mfcc'].apply(lambda x : np.pad(x, ((0,0),(0, longest - x.shape[1])), mode='constant'))

        self.data = data

    def __getitem__(self, index):

        row = self.data.iloc[index]

        features = row['mfcc']
        length = row['length']
        target = row['target']

        return (torch.FloatTensor(features.T), length),  target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def args(parser):

        parser.add_argument('--data_path')


        return super(AudioDataset, AudioDataset).args(parser)