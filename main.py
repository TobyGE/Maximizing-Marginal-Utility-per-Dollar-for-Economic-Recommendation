import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
from heapq import heappush, heappop
from torch.utils.data import Dataset, DataLoader

import evaluation
import data_loader
from model import MF, MUD

def main(category):
	params = dict()
	params['lr'] = 0.001
	params['batch_size'] = 16
	params['epoch_limit'] = 1
	params['w_decay'] = 1
	params['negNum'] = 1000
	params['epsilon'] = 1e-4

	params['l_size'] = 32
	params['gpu']=False

	train, val, test = data_loader.read_data(category)
	item_price = data_loader.get_price(category)
	item_related = data_loader.read_related(category)
	distribution = data_loader.get_distribution(category)
	trainset = data_loader.TransactionData(train, item_related, \
				item_price, distribution)
	valset = data_loader.UserTransactionData(val, item_price, \
				trainset.itemNum, trainset.userHist)
	testset = data_loader.UserTransactionData(test, item_price, \
				trainset.itemNum, trainset.userHist)

	Rating = MF(userLen = trainset.userNum, itemLen = trainset.itemNum,\
			 params = params)
	optimizer_r = opt.SGD(Rating.parameters(), lr = params['lr'], \
			weight_decay = params['w_decay'])
	
	criterion_rating = nn.MSELoss(reduction = 'sum')
	criterion_risk = nn.MSELoss(reduction = 'sum')
	criterion_MUD = nn.LogSoftmax()

	trainset.set_negN(4)
	trainLoader = DataLoader(trainset, batch_size = params['batch_size'], \
							shuffle = True, num_workers = 0)
	valset.set_negN(1000)
	valLoader = DataLoader(valset, batch_size = 1, \
							shuffle = True, num_workers = 0)
	testset.set_negN(1000)
	testLoader = DataLoader(testset, batch_size = 1, \
							shuffle = True, num_workers = 0)

	epsilon = params['epsilon']
	epoch = 1
	error = np.float('inf')

	print("starting pretrain the rating matrix...")
	while epoch < params['epoch_limit']:
		runningLoss = []
		epoch += 1
		print("Epoch " + str(epoch) + " training...")
		L = len(trainLoader.dataset)
		pbar = tqdm(total=L)
		for i, batchData in enumerate(trainLoader):
			optimizer_r.zero_grad()
			# get input
			users = torch.LongTensor(batchData['user']).to(Rating.device)
			items = torch.LongTensor(batchData['item']).to(Rating.device)
			pre_r = Rating.forward(users, items)
			r = torch.FloatTensor(batchData['rating']).to(Rating.device)

			loss = criterion_rating(pre_r, r)
			loss.backward()
			optimizer_r.step()
			runningLoss.append(loss.item())

			pbar.update(users.shape[0])
		pbar.close()
		meanMSE = np.mean(np.array(runningLoss))
		improvement = np.abs(error - meanMSE)
		error = meanMSE
		if improvement <= epsilon:
			print('pre-train stop early')
			break

	model = MUD(userLen = trainset.userNum, itemLen = trainset.itemNum,\
			 distribution = distribution, item_price = item_price, \
			 RMF = Rating, params = params)
	optimizer = opt.SGD(model.parameters(), lr = params['lr'], \
			weight_decay = params['w_decay'])

	epsilon = params['epsilon']
	epoch = 0
	error = np.float('inf')

	trainErrorList = []
	valErrorList = []
	valHistory = []
	explodeTempreture = 3
	convergenceTempreture = 3
	print("starting ROM model")
	while epoch < params['epoch_limit']:
		epoch += 1
		print("Epoch " + str(epoch) + " training...")
		L = len(trainLoader.dataset)
		pbar = tqdm(total = L)
		for i, batchData in enumerate(trainLoader):
			optimizer.zero_grad()
			# get input
			users = torch.LongTensor(batchData['user']).to(model.device)
			# print(users.shape)
			items = torch.LongTensor(batchData['item']).to(model.device)
			# print(items.shape)
			negItems = torch.LongTensor(batchData['negItem']).to(model.device)
			# print(negItems.shape)
			nusers = users.view(-1,1).expand(users.shape[0], negItems.shape[1])

			pOut = model.forward(users,items)
			nOut = model.forward(nusers, negItems)
			totalOut = torch.cat((pOut,nOut),1)

			loss = criterion_MUD(totalOut)[:,0]
			loss.backward()
			optimizer.step()
			pbar.update(users.shape[0])
		pbar.close()
		print("Epoch " + str(epoch) + " training risk...")
		pbar = tqdm(total = L)
		for i, batchData in enumerate(trainLoader):
			optimizer.zero_grad()
			users = torch.LongTensor(batchData['user']).to(model.device)
			items = torch.LongTensor(batchData['user']).to(model.device)

			eu = model.EU(users,items)
			ue = model.UE(users,items)

			loss = criterion_risk(ue, eu)
			loss.backward()
			optimizer.step()
			pbar.update(users.shape[0])
		pbar.close()

		#validation
		print("Epoch " + str(epoch) + " validating...")
		L = len(valLoader.datset)
		pbar = tqdm(total = L)
		model.eval()
		with torch.no_grad():
			scoreDict = dict()
			for i, batchData in enumerate(valLoader):
				user = torch.LongTensor(batchData['user']).to(model.device)
				posItems = torch.LongTensor(batchData['posItem']).to(model.device)
				negItems = torch.LongTensor(batchData['negItem']).to(model.device)
				budget = torch.FloatTensor(batchData['budget']).to(model.device)
				posPrices = torch.FloatTensor(batchData['posPrice']).to(model.device)
				negPrices = torch.FloatTensor(batchData['negPrice']).to(model.device)

				items = torch.cat((posItems, negItems),1).view(-1)
				prices = torch.cat((posPrices, negPrices),1).view(-1)
				users = user.expand(items.shape[0])

				out = model.forward(users,items)
				scoreHeap = list()
				for j in range(out.shape[0]):
					gt = False
					if j < posItems.shape[1]:
						gt = True
					if prices[j] > budget:
						continue
					heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
				scores = list()
				candidate = len(scoreHeap)
				for k in range(candidate):
					scores.append(heappop(scoreHeap))
				pbar.update(1)
				scoreDict[user[0]] = (scores, posItems.shape[1])
		pbar.close()

		valHistory.append(evaluation.ranking_performance(scoreDict,100))
		valError = 1 - valHistory[-1]["avg_ndcg"][0]
		valErrorList.append(valError)
		improvement = np.abs(error - valError)
		error = valError
		if improvement < epsilon:
			break

		# test
		print("starting test...")
		with torch.no_grad():
			scoreDict = dict()
			for i, batchData in enumerate(testLoader):
				user = torch.LongTensor(batchData['user']).to(model.device)
				posItems = torch.LongTensor(batchData['posItem']).to(model.device)
				negItems = torch.LongTensor(batchData['negItem']).to(model.device)
				budget = torch.FloatTensor(batchData['budget']).to(model.device)
				posPrices = torch.FloatTensor(batchData['posPrice']).to(model.device)
				negPrices = torch.FloatTensor(batchData['negPrice']).to(model.device)

				items = torch.cat((posItems, negItems),1).view(-1)
				prices = torch.cat((posPrices, negPrices),1).view(-1)
				users = user.expand(items.shape[0])

				out = model.forward(users,items)
				scoreHeap = list()
				for j in range(out.shape[0]):
					gt = False
					if j < posItems.shape[1]:
						gt = True
					if prices[j] > budget:
						continue
					heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
				scores = list()
				candidate = len(scoreHeap)
				for k in range(candidate):
					scores.append(heappop(scoreHeap))
				pbar.update(1)
				scoreDict[user[0]] = (scores, posItems.shape[1])
		pbar.close()
		testResult = evaluation.ranking_performance(scoreDict,100)


if __name__ == '__main__':
	main("Baby")

