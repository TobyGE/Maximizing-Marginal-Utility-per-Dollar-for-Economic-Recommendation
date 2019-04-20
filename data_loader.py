import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import torch.nn as nn

def read_data(category):
	address = "./data/" + category + "/" + category + "_"
	with open(address + "TrainSamples.txt","r") as f:
		data = f.readlines()
	TrainSamples = []
	for line in data:
		row = line[:-1].split(",")
		sample = [int(i) for i in row]
		TrainSamples.append(sample)

	with open(address + "ValidationSamples.txt","r") as f:
		data = f.readlines()
	ValSamples = []
	for line in data:
		row = line[:-1].split(",")
		sample = [int(i) for i in row]
		ValSamples.append(sample)

	with open(address + "TestSamples.txt","r") as f:
		data = f.readlines()
	TestSamples = []
	for line in data:
		row = line[:-1].split(",")
		sample = [int(i) for i in row]
		TestSamples.append(sample)

	return TrainSamples, ValSamples, TestSamples

def get_price(category):
	address = "./data/" + category + "/" + category + "_" + "item_price.npy"
	price = np.load(address)
	return price

def read_related(category):
	address = "./data/" + category + "/" + category + "_"
	with open(address + "related_index.json","r") as f:
		related = json.load(f)
	return related

def get_distribution(category):
	address = "./data/" + category + "/" + category + "_" + "ItemResult.npy"
	distribution = np.load(address)
	return distribution

class TransactionData(Dataset):
	"""docstring for TransactionData"""
	def __init__(self, transctions, related, item_price, rating_distribution):
		super(TransactionData, self).__init__()
		self.transctions = transctions
		self.related = related
		self.L = len(transctions)
		self.users = np.unique(np.array(transctions)[:,0])
		self.userNum = len(self.users)
		self.itemNum = len(related)
		self.negNum = 2
		self.item_price = item_price
		self.rating_distribution = rating_distribution
		self.userHist = [[] for i in range(self.userNum)]
		for row in transctions:
			self.userHist[row[0]].append(row[1])
		

	def __len__(self):
		return self.L

	def __getitem__(self,idx):
		row = self.transctions[idx]
		user = row[0]
		item = row[1]
		rating = row[2]
		price = self.item_price[item]
		negItem = self.get_neg(user, item)
		negPrice = []
		distribution = self.rating_distribution[item]
		for i in negItem:
			negPrice.append(self.item_price[i])
		return {"user": torch.tensor(user).to(torch.long), \
				"item": torch.tensor(item).to(torch.long), \
				"r_distribution": torch.tensor(distribution).to(torch.float), \
				"price": torch.tensor(price).to(torch.float), \
				"rating": torch.tensor(rating).to(torch.float), \
				"negItem": torch.tensor(negItem).to(torch.long), \
				"negPrice": torch.tensor(negPrice).to(torch.float)}

	def get_neg(self, userid, itemid):
		neg = self.related[str(itemid)]
		hist = self.userHist[userid]
		for k in neg:
			if k in hist:
				neg.remove(k)
		if len(neg) < self.negNum:
			for i in range(self.negNum - len(neg)):
				while True:
					negId = np.random.randint(self.itemNum)
					if negId not in hist and negId not in neg:
						neg.append(negId)
						break
		else:
			neg = random.sample(neg, self.negNum)
		return neg

	def set_negN(self, n):
		if n < 1:
			return
		self.negNum = n

class UserTransactionData(Dataset):
	"""docstring for UserTransactionData"""
	def __init__(self, transactions, item_price, itemNum, trainHist):
		super(UserTransactionData, self).__init__()
		self.transactions = transactions
		self.user = np.unique(np.array(transactions)[:,0])
		self.L = len(self.user)
		self.userNum = self.L
		self.itemNum = itemNum
		self.negNum = 1000
		self.userHist = [[] for i in range(self.userNum)]
		self.trainHist = trainHist
		for row in transactions:
			self.userHist[row[0]].append(row[1])

	def __len__(self):
		return self.L

	def __getitem__(self, idx):
		user = self.user[idx]
		posItem = self.userHist[idx]
		posPrice = []
		for i in posItem:
			posPrice.append(self.item_price[i])
		negPrice = []
		negItem = self.get_neg(idx)
		for i in negItem:
			negPrice.append(self.item_price[i])
		budget = self.get_budget(idx)
		return {"user": torch.tensor(user).to(torch.long), \
				"budget": torch.tensor(budget).to(torch.float), \
				"posItem": torch.tensor(posItem).to(torch.long), \
				"posPrice": torch.tensor(posPrice).to(torch.float), \
				"negPrice": torch.tensor(negPrice).to(torch.float), \
				"negItem": torch.tensor(negItem).to(torch.long)}

	def get_neg(self, userId):
		hist = self.userHist[userId] + self.trainHist[userId]
		neg = []
		for i in range(self.negNum):
			while True:
				negId = np.random.randint(self.itemNum)
				if negId not in hist and negId not in neg:
					neg.append(negId)
					break
		return neg

	def set_negN(self, n):
		if n < 1: 
			return
		self.negNum = n

	def get_budget(self, userId):
		price = []
		for i in self.trainHist[userIf]:
			price = self.item_price[i]
		budget = np.max(np.array(price))
		return budget
		

# if __name__ == '__main__':
	# train_data, val_data, test_data = read_data("Baby")
	# related = read_related("Baby")
	# TrainTransaction = TransactionData(train_data, related)
	# print(TrainTransaction[0])
	# TestTransaction = UserTransactionData(test_data)
	# print(TestTransaction[0])
	# print(nn.Embedding(2,3)(torch.tensor(1)))

