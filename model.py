
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
        uE = self.uEmbed(users)
        uB = self.uBias(users)
        iE = self.itemEmbed(items)
        iB = self.itemBias(items)
        try: 
            gB = self.globalBias.weight.data.expand(users.shape[0],users.shape[1])
            score = gB + uB + iB + torch.mul(uE, iE).sum(2)
        except:
            gB = self.globalBias.weight.data.expand(users.shape[0],1)
            score = gB + uB + iB + torch.mul(uE, iE).sum(1).view(-1,1)
        return score

class MUD(nn.Module):
    """docstring for MUD"""
    def __init__(self, userLen, itemLen, distribution, item_price, RMF, params):
        super(MUD, self).__init__()
        self.userNum = userLen
        self.itemNum = itemLen
        self.distribution = torch.tensor(distribution).to(torch.float)
        self.price = torch.tensor(item_price).to(torch.float)
        self.rating = RMF
        self.params = params
        if 'gpu' in params and params['gpu'] == True:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        l_size = params['l_size']

        self.gBias = nn.Embedding(1,1)
        self.uBias = nn.Embedding(userLen,1)
        self.itemBias = nn.Embedding(itemLen,1)
        self.uEmbed = nn.Embedding(userLen, l_size)
        self.itemEmbed = nn.Embedding(itemLen, l_size) 

    def forward(self, users, items):
        uE = self.uEmbed(users)
        uB = self.uBias(users)
        iE = self.itemEmbed(items)
        iB = self.itemBias(items)
#         print("size")
        
        # print(uE.shape)
        # print(uB.shape)
        # print(iE.shape)
        # print(iB.shape)
        try:
            gB = self.gBias.weight.data.expand(users.shape[0],users.shape[1])
            alpha = gB + uB + iB + torch.mul(uE, iE).sum(2)
        except:
            gB = self.gBias.weight.data.expand(users.shape[0],1)
#             print(torch.mul(uE,iE).sum(1).shape)
            alpha = gB + uB + iB + torch.mul(uE, iE).sum(1).view(-1,1)
        with torch.no_grad():
            r = self.rating.forward(users, items)
            tanh_r = torch.tanh(r).view(-1,1)
        price = self.price[items]
#         print(gB.shape)
#         print("alpha")
#         print(alpha.shape)
#         print(price.shape)
        u = torch.mul(alpha, tanh_r)
#         print(u.shape)
        out = 0.5 * torch.div(u.view(-1), torch.sigmoid(price))
#         print("out")
#         print(out.shape)
        return out

    def EU(self, users, items):
        gB = self.gBias.weight.data.expand(users.shape[0],1)
        uE = self.uEmbed(users)
        uB = self.uBias(users)
        iE = self.itemEmbed(items)
        iB = self.itemBias(items)
        
        alpha = gB + uB + iB + torch.mul(uE, iE).sum(1).view(-1,1).expand(users.shape[0],5)
        distribution = self.distribution[items]
        rating = torch.tensor([1,2,3,4,5]).expand(users.shape[0],5).to(torch.float)
        tanh_r = torch.tanh(rating)
        U = torch.log(torch.tensor(2).to(torch.float)) * torch.mul(alpha, tanh_r)
        EU = torch.mul(distribution, U).sum(1)
        return EU

    def UE(self, users, items):
        gB = self.gBias.weight.data.expand(users.shape[0],1)
        uE = self.uEmbed(users)
        uB = self.uBias(users)
        iE = self.itemEmbed(items)
        iB = self.itemBias(items)
        
        alpha = gB + uB + iB + torch.mul(uE, iE).sum(1).view(-1,1)
        distribution = self.distribution[items]
        rating = torch.tensor([1,2,3,4,5]).expand(users.shape[0],5).to(torch.float)
        r_bar = torch.mul(distribution, rating).sum(1)
        tanh_r_bar = torch.tanh(r_bar)
        UE = torch.log(torch.tensor(2).to(torch.float)) * torch.mul(alpha, tanh_r_bar)
        return UE

        
        