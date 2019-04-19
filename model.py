
import numpy as np
import torch
import torch.nn as nn
import data_loader
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm



class MF(nn.Module):
    """
        - userLen: the number of users
        - itemLen: the number of items
        - params: the parameters dict used for constructing model
            - l_size: latent dimension size
            - gpu: True/False, whether using GPU
            
    """
    def __init__(self, userLen, itemLen, params):
        super(MF, self).__init__()
        self.userNum = userLen
        self.itemNum = itemLen
        self.params = params
        if 'gpu' in params and params['gpu'] == True:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        l_size = params['l_size']
        
        """
            Initialize  global bias,
                        user bias,
                        item bias,
                        user embedding,
                        item embedding
        """
        self.globalBias = nn.Embedding(1,1)
        self.uBias = nn.Embedding(userLen,1)
        self.itemBias = nn.Embedding(itemLen,1)
        self.uEmbed = nn.Embedding(userLen, l_size)
        self.itemEmbed = nn.Embedding(itemLen, l_size)
    
    def forward(self, users, items):
        gB = self.globalBias.weight.data.expand(users.shape[0],1)
        uE = self.uEmbed(users)
        uB = self.uBias(users)
        iE = self.itemEmbed(items)
        iB = self.itemBias(items)
        
        score = gB + uB + iB + torch.mul(uE, iE).sum(1).view(-1,1) 
        return score
        