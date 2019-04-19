import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
from heapq import heappush, heappop

import evaluation
import data_loader
import model

def main(category):
	params = dict()
	params['lr'] = 0.001
	params['batch_size'] = 16
	params['epoch_limit'] = 10
	params['weight_decay'] = 1
	params['negNum'] = 1000

	params['l_size'] = 32
	params['gpu']=False

	train, val, test = data_loader.read_data(category)
	item_price = data_loader.get_price(category)
	item_related = data_loader.read_related(category)
	trainset = data_loader.TransactionData(train, item_related, item_price)
	valset = data_loader.UserTransactionData(val, item_price, \
				trainset.itemNum, trainset.userHist)


if __name__ == '__main__':
	main("Baby")

