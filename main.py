import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from heapq import heappush, heappop
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import evaluation
import data_loader
from model import MF, MUD
import pdb

def main(category):
    params = dict()
    params['lr'] = 5e-4
    params['batch_size'] = 128
    params['epoch_limit'] = 3
    params['w_decay'] = 1
    params['negNum_test'] = 1000
    params['epsilon'] = 1e-2
    params['negNum_train'] = 4
    params['l_size'] = 100
    params['gpu']= False

    train, val, test = data_loader.read_data(category)
    item_price = data_loader.get_price(category)
    item_related = data_loader.get_related(category)
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
    criterion_MUD = nn.LogSoftmax(dim=0)

    trainset.set_negN(params['negNum_train'])
    trainLoader = DataLoader(trainset, batch_size = params['batch_size'], \
                            shuffle = True, num_workers = 0)
    valset.set_negN(params['negNum_test'])
    valLoader = DataLoader(valset, batch_size = 1, \
                            shuffle = True, num_workers = 0)
    testset.set_negN(params['negNum_test'])
    testLoader = DataLoader(testset, batch_size = 1, \
                            shuffle = True, num_workers = 0)

    epsilon = params['epsilon']
    epoch = 0
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
            users = torch.LongTensor(batchData['user'])#.to(Rating.device)
            items = torch.LongTensor(batchData['item'])#.to(Rating.device)
            pre_r = Rating.forward(users, items)
            r = torch.FloatTensor(batchData['rating'])#.to(Rating.device)

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
#             print(users.shape)
            items = torch.LongTensor(batchData['item']).to(model.device)
#             print(items.shape)
            negItems = torch.LongTensor(batchData['negItem']).reshape(-1).to(model.device)
#             print(negItems.shape)
            
            nusers = users.view(-1,1) 
            nusers = nusers.expand(nusers.shape[0], params['negNum_train']).reshape(-1)
#             print(nusers.shape)
            
            
            pOut = model.forward(users,items).view(-1,1)
#             print(pOut.shape)
            
            nOut = model.forward(nusers, negItems)
            nOut = nOut.reshape(-1,params["negNum_train"])
#             print(nOut.shape)
            
            totalOut = torch.cat((pOut,nOut),dim=1)
#             print (totalOut.shape)
            
            loss = torch.mean(criterion_MUD(totalOut)[:,0])
#             print(loss.shape)
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
#         print("Epoch " + str(epoch) + " validating...")
#         L = len(valLoader.dataset)
#         pbar = tqdm(total = L)
#         model.eval()
#         with torch.no_grad():
#             scoreDict = dict()
#             for i, batchData in enumerate(valLoader):
#                 user = torch.LongTensor(batchData['user'])#.to(model.device)
#                 posItems = torch.LongTensor(batchData['posItem'])#.to(model.device)
#                 negItems = torch.LongTensor(batchData['negItem'])#.to(model.device)
#                 budget = torch.FloatTensor(batchData['budget'])#.to(model.device)
#                 posPrices = torch.FloatTensor(batchData['posPrice'])#.to(model.device)
#                 negPrices = torch.FloatTensor(batchData['negPrice'])#.to(model.device)

#                 items = torch.cat((posItems, negItems),1).view(-1)
#                 prices = torch.cat((posPrices, negPrices),1).view(-1)
#                 users = user.expand(items.shape[0])

#                 out = model.forward(users,items)
#                 scoreHeap = list()
#                 for j in range(out.shape[0]):
#                     gt = False
#                     if j < posItems.shape[1]:
#                         gt = True
                    # if prices[j] > budget:
                    #     heappush(scoreHeap, (100, (0 + items[j].cpu().numpy(), gt)))
                    # else:
                    #     heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
#                 scores = list()
#                 candidate = len(scoreHeap)
#                 for k in range(candidate):
#                     scores.append(heappop(scoreHeap))
#                 pbar.update(1)
#                 scoreDict[user[0]] = (scores, posItems.shape[1])
#         pbar.close()

#         valHistory.append(evaluation.ranking_performance(scoreDict,10))
#         valError = 1 - valHistory[-1]["avg_ndcg"][0]
#         valErrorList.append(valError)
#         improvement = np.abs(error - valError)
#         error = valError
#         if improvement < epsilon:
#             print("stop early")
#             break

    # test
    print("starting test...")
    L = len(testLoader.dataset)
    pbar = tqdm(total = L)
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
                    heappush(scoreHeap, (1000, (0 + items[j].cpu().numpy(), gt)))
                else:
                    heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
                # heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
            scores = list()
            candidate = len(scoreHeap)
            for k in range(candidate):
                scores.append(heappop(scoreHeap))
            pbar.update(1)
            scoreDict[user[0]] = (scores, posItems.shape[1])
    pbar.close()
    testResult = evaluation.ranking_performance(scoreDict,10)

def CF(category):
    params = dict()
    params['lr'] = 1e-2
    params['negNum_train'] = 3
    params["negNum_test"] = 1000
    params['epoch_limit'] = 3
    params['w_decay'] = 1.
    params['batch_size'] = 128
    params['gpu'] = False
    params['l_size'] = 16
    params['epsilon'] = 0.01

    train, val, test = data_loader.read_data(category)
    item_price = data_loader.get_price(category)
    item_related = data_loader.get_related(category)
    distribution = data_loader.get_distribution(category)
    trainset = data_loader.TransactionData(train, item_related, \
                item_price, distribution)
    testset = data_loader.UserTransactionData(test, item_price, \
                trainset.itemNum, trainset.userHist)
    avg_rating = trainset.get_avgRating()

    model = MF(userLen = trainset.userNum, itemLen = trainset.itemNum,\
            avg_rating = avg_rating, params = params)
    optimizer = opt.SGD(model.parameters(), lr = params['lr'], weight_decay = params['w_decay'])
    criterion = nn.MSELoss()

    trainset.set_negN(params['negNum_train'])
    trainLoader = DataLoader(trainset, batch_size = params['batch_size'], \
            shuffle = True, num_workers = 0)
    testset.set_negN(params["negNum_test"])
    testLoader = DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 4)

    runningLoss = []
    printLoss = []
    epoch = 0
    while epoch < params['epoch_limit']:
        epoch += 1
        L = len(trainLoader.dataset)
        pbar = tqdm(total = L)
        for i, batchData in enumerate(trainLoader):
            optimizer.zero_grad()
            # get input
            users = torch.LongTensor(batchData['user'])#.to(Rating.device)
            items = torch.LongTensor(batchData['item'])#.to(Rating.device)
            pre_r = model.forward(users, items)
            r = torch.FloatTensor(batchData['rating'])#.to(Rating.device)

            loss = criterion(pre_r, r)
            runningLoss.append(loss.item())
            if (i+1) % 50 == 0:
                printLoss.append(np.mean(np.array(runningLoss[-50:])))
            loss.backward()
            optimizer.step()
            pbar.update(users.shape[0])
        pbar.close()
        with torch.no_grad():
            L = len(testLoader.dataset)
            pbar = tqdm(total = L)
            scoreDict = dict()
            for i, batchData in enumerate(testLoader):
                if i > 1000:
                    break
                user = torch.LongTensor(batchData['user']).to(model.device)
                posItems = torch.LongTensor(batchData['posItem']).to(model.device)
                negItems = torch.LongTensor(batchData['negItem']).to(model.device)
                items = torch.cat((posItems, negItems),1).view(-1)
                users = user.expand(items.shape[0])
                out = model.forward(users,items)
                # print("user: \n" + str(user))
                # print("posItems: \n" + str(posItems))
                # print("negItems: \n" + str(negItems))
                # print("items: \n" + str(items))
                # print("users: \n" + str(users))
                # print("out: \n" + str(out))
                # print("model: \n" + str(model.))
                # input()
                # pdb.set_trace()
                scoreHeap = list()
                for j in range(out.shape[0]):
                    gt = False
                    if j < posItems.shape[1]:
                        gt = True
                    # if prices[j] > budget:
                    #     heappush(scoreHeap, (100, (0 + items[j].cpu().numpy(), gt)))
                    # else:
                    #     heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
                    heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
                scores = list()
                candidate = len(scoreHeap)
                for k in range(candidate):
                    scores.append(heappop(scoreHeap))
                pbar.update(1)
                scoreDict[user[0]] = (scores, posItems.shape[1])
            pbar.close()
        testResult = evaluation.ranking_performance(scoreDict,10)
    with torch.no_grad():
        L = len(testLoader.dataset)
        pbar = tqdm(total = L)
        scoreDict = dict()
        for i, batchData in enumerate(testLoader):
            user = torch.LongTensor(batchData['user']).to(model.device)
            posItems = torch.LongTensor(batchData['posItem']).to(model.device)
            negItems = torch.LongTensor(batchData['negItem']).to(model.device)
            items = torch.cat((posItems, negItems),1).view(-1)
            users = user.expand(items.shape[0])
            out = model.forward(users,items)
            scoreHeap = list()
            for j in range(out.shape[0]):
                gt = False
                if j < posItems.shape[1]:
                    gt = True
                # if prices[j] > budget:
                #     heappush(scoreHeap, (100, (0 + items[j].cpu().numpy(), gt)))
                # else:
                #     heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
                heappush(scoreHeap, (1 - out[j].cpu().numpy(), (0 + items[j].cpu().numpy(), gt)))
            scores = list()
            candidate = len(scoreHeap)
            for k in range(candidate):
                scores.append(heappop(scoreHeap))
            pbar.update(1)
            scoreDict[user[0]] = (scores, posItems.shape[1])
        pbar.close()
    testResult = evaluation.ranking_performance(scoreDict,10)


if __name__ == '__main__':
    # main("Baby")
    CF("Baby")

