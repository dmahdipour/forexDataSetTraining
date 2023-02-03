import torch
import pandas as pd
import numpy as np


class myDataset(torch.utils.data.Dataset):
    def __init__(self):
        data = pd.read_csv('data-GBPUSD.csv', header=0)
        self.x = data.values[:-1 , 1:]
        self.y = data.values[1:,2]
        self.maxX = np.max(self.x, axis=0)
        self.minX = np.min(self.x, axis=0)
        self.maxY = np.max(self.y, axis=0)
        self.minY = np.min(self.y, axis=0)

        self.normalX = (self.x-self.minX)/self.maxX
        self.normalY = (self.y-self.minY)/self.maxY

        self.normalX = self.normalX.astype('float32')
        self.normalY = self.normalY.astype('float32')
        

    def __getitem__(self, index):
        return (self.normalX[index], self.normalY[index])

    def __len__(self):
        return len(self.x)


ds = myDataset()
print(ds[0])
