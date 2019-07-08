import numpy as np
import math
import json
import copy
import random
import time
import matplotlib.pyplot as plt
import pickle

def read_data(category):
    path = "./data/" + category + "/" + category
    train_sample_list = open(path + "_TrainSamples.txt",'r')
    train_samples = []
    for i in train_sample_list:
        s = i.split(",")
        one_sample = [int(s[0]), int(s[1]), int(s[2][0])]
        train_samples.append(one_sample)
        
    validation_sample_list = open(path + "_ValidationSamples.txt",'r')
    validation_samples = []
    for i in validation_sample_list:
        s = i.split(",")
        one_sample = [int(s[0]), int(s[1]), int(s[2][0])]
        validation_samples.append(one_sample)
        
    test_sample_list = open(path + "_TestSamples.txt",'r')
    test_samples = []
    for i in test_sample_list:
        s = i.split(",")
        one_sample = [int(s[0]), int(s[1]), int(s[2][0])]
        test_samples.append(one_sample)
        
    return train_samples, validation_samples, test_samples

class MUD():
    def __init__(self, samples, num_users, num_items, CFK, CFlr, CFbeta, CFiterations, K, lr, beta, iterations, item_path, sample_path,item_price_path):
        """
        CFlr: learning rate for CF
        CFK: latent dimension in CF
        CFbeta: the regularizer of CF
        CFiterations: the number of iterations of CF
        lr: learning rate
        K: latent dimension
        beta: regularizer
        iterations: the number of iterations
        item_path: the path for rating distribution
        sample_path: the path for related items
        item_price_path: the path for the price of items
        """
        self.samples = samples
        self.num_users = num_users
        self.num_items = num_items
        self.CFK = CFK
        self.CFlr = CFlr
        self.CFbeta = CFbeta
        self.CFiterations = CFiterations
        self.K = K
        self.lr = lr
        self.beta = beta
        self.iterations = iterations
        self.item_path = item_path
        self.sample_path = sample_path
        self.item_price_path = item_price_path
        
    def train(self):
        
        """
        initialize the parameters
        """
        ### Initialize user and item latent feature matrice of alpha
        self.Pa = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Qa = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        ### Initialize the biases of alpha
        self.ba_u = np.zeros(self.num_users)
        self.ba_i = np.zeros(self.num_items)
        self.ba = 0
        
        ### Initialize user and item latent feature matrice of rating
        self.Pr = np.random.normal(scale=1./self.CFK, size=(self.num_users, self.K))
        self.Qr = np.random.normal(scale=1./self.CFK, size=(self.num_items, self.K))
        
        ### Initialize the biases of rating
        self.br_u = np.zeros(self.num_users)
        self.br_i = np.zeros(self.num_items)
        rating = [i[2] for i in self.samples]
        self.br = np.mean(rating)
        
        
        all_bought = self.get_bought()
        
        with open(self.sample_path, 'r') as f:
             negative_samples= json.load(f)
        
        ### the rating distribution
        item_result = self.get_item_result()
        
        ### the price of items
        item_price = np.load(self.item_price_path)
                
        
        n = 0
        print("===========================")
        mse = self.mse()
        print("Iteration: %d ; error = %.4f" % (n, mse))
        n += 1
        
        while(n<=self.CFiterations):
            print("===========================")
            self.cf()
            mse = self.mse()
            print("Iteration: %d ; error = %.4f" % (n, mse))
            n += 1
        n = 1
        
        while(n<=self.iterations):
            print("===========================")
            self.risk(item_price, item_result)
            self.sgd(negative_samples, all_bought, item_result,item_price)
            print("Iteration: %d" % n)
            n += 1
        
      
    def cf(self):
        """
        get predicted rating
        """
        for i,j,q in self.samples:
            rij = self.get_rating(i,j)
            commonTerm_r1 = - 2 * (q - rij)
            self.br_u[i] -= self.CFlr * (commonTerm_r1 + 2 * self.CFbeta * self.br_u[i])
            self.br_i[j] -= self.CFlr * (commonTerm_r1 + 2* self.CFbeta * self.br_i[j])
            self.Pr[i, :] -= self.CFlr * (commonTerm_r1 * self.Qr[j, :] + 2 * self.CFbeta * self.Pr[i,:])
            self.Qr[j, :] -= self.CFlr * (commonTerm_r1 * self.Pr[i, :] + 2 * self.CFbeta * self.Qr[j,:])
        
    def risk(self, item_price, item_result):
        """
        risk neutral
        """
        for i, j, q in self.samples:
            aij = self.get_aij(i,j)
            commonTerm_a1 = 2 * (aij * self.u_bar(item_result[j]) * math.log(2) - 
                                aij * self.u_r_bar(item_result[j]) * math.log(2)
                               ) * math.log(2) * (self.u_bar(item_result[j]) - self.u_r_bar(item_result[j]))
            self.ba_u[i] -= self.lr * (commonTerm_a1 + 2 * self.beta * self.ba_u[i])
            self.ba_i[j] -= self.lr * (commonTerm_a1 + 2 * self.beta * self.ba_i[j])
            self.ba -= self.lr * (commonTerm_a1 + 2 * self.beta * self.ba)
            self.Pa[i, :] -= self.lr * (commonTerm_a1 * self.Qa[j, :] + 2 * self.beta * self.Pa[i,:])
            self.Qa[j, :] -= self.lr * (commonTerm_a1 * self.Pa[i, :] + 2 * self.beta * self.Qa[j,:])  
        
        
    def sgd(self,negative_samples, all_bought, item_result,item_price):
        """
        maximize mud
        """
        all_items = [x for x in range(self.num_items)]
        for i, j, q in self.samples:
            negative_samples_j = negative_samples[str(j)]
            bought = all_bought[i]
            for k in bought:
                if k in negative_samples_j:
                    negative_samples_j.remove(k) 
            negative_samples_j.append(j)
            if len(negative_samples_j) < 2:
                not_bought = list(set(all_items) - set(bought))
                negative_samples_j.append(random.sample(not_bought,1)[0])
            
            aij = self.get_aij(i,j)
            rij = self.get_rating(i,j)
            sigij = 2/(1+math.exp(-rij)) - 1
            sigijd = 2 * math.exp(-rij) / np.square(1 + math.exp(-rij))
            sigpij = 1/(1+math.exp(-item_price[j]))
            
            commonTerm_a = sigij / sigpij
            temp_0 = 0
            temp_1 = 0
            temp_2 = 0
            aik_total = []
            rik_total = []
            for k in negative_samples_j:
                k = int(k)
                aik = self.get_aij(i,k)
                aik_total.append(aik)
                rik = self.get_rating(i,k)
                rik_total.append(rik)
                sigik = 2/(1+math.exp(-rik)) - 1
                sigpik = 1/(1+math.exp(-item_price[k]))
                temp_0 += math.exp(aik * sigik / sigpik)
                temp_1 += math.exp(aik * sigik / sigpik) * sigik / sigpik
                
            commonTerm_ai = commonTerm_a - temp_1/temp_0
            commonTerm_aj = commonTerm_a - commonTerm_a * math.exp(aij * commonTerm_a)/temp_0
            
            self.ba_u[i] += self.lr * (commonTerm_ai - 2 * self.beta * self.ba_u[i])
            self.ba_i[j] += self.lr * (commonTerm_aj - 2 * self.beta * self.ba_i[j])
            self.ba += self.lr * (commonTerm_ai - 2 * self.beta * self.ba)
            self.Pa[i, :] += self.lr * (commonTerm_ai * self.Qa[j, :] - 2 * self.beta * self.Pa[i,:])
            self.Qa[j, :] += self.lr * (commonTerm_aj * self.Pa[i, :] - 2 * self.beta * self.Qa[j,:])

    
    def u_r_bar (self, row):
        """
        calculate the utility of expected rating
        """
        mean_val = 0
        for i in range(len(row)):
            mean_val += row[i]*(i+1)       
        val = 2/(1+math.exp(-mean_val))-1
        return val
    
    def u_bar(self,row):
        """
        calculate the expectation of the utility
        """
        val = 0
        for i in range(len(row)):
            val += row[i]*(2/(1+math.exp(-i-1))-1)
        return val
    
    def get_bought(self):
        bought = [[] for i in range(self.num_users)]
        for s in self.samples:
            bought[s[0]].append(s[1])
        print("computing bought finished")
        return bought
    
    def get_item_result(self):
        item_result = np.load(self.item_path)
        return item_result
    
    def mse(self):
        error = 0
        for s in self.samples:
            rij = self.get_rating(s[0],s[1])
            error += pow(rij - s[2], 2)
        return np.sqrt(error/len(self.samples))
        
    def get_aij(self,i,j):
        aij = self.ba + self.ba_u[i] + self.ba_i[j] + self.Pa[i, :].dot(self.Qa[j, :].T)
        return aij
    
    def get_rating(self,i,j):
        rij = self.br + self.br_u[i] + self.br_i[j] + self.Pr[i, :].dot(self.Qr[j, :].T)
        return rij
    
def TrainTest(category,u_num,i_num,params):
    path = "./data/" + category + "/" + category
    train_samples,val_samples,test_samples = read_data(category)
    mud = MUD(train_samples, num_users=u_num, num_items=i_num, 
        CFK=params['CFK'],CFlr=params['CFlr'],CFbeta=params['CFbeta'],CFiterations=params['CFiterations'],
        K=params['K'],lr=params['lr'], beta=params['beta'], iterations=params['iterations'],
        item_path=path+'_ItemResult.npy',
        sample_path=path+'_related_index.json',
        item_price_path=path+'_item_price.npy'
       )
    
    training_process = mud.train()
    
    item_price = np.load(path+'_item_price.npy')
    
    total_price = [[] for i in range(u_num)]
    for s in train_samples:
        total_price[s[0]].append(item_price[s[1]])
    budget = []
    for i in range(u_num):
        b = max(total_price[i])
        budget.append(b)
        
    test_list = open(path+"_TestList.txt",'r')
    test_data = []
    for td in test_list:
        p = td[1:-2].split(',')
        one_test_data = []
        for s in p:
            one_test_data.append(int(s))
        test_data.append(one_test_data)
        
    print("start testing...")
    right_num = np.zeros([10,1])
    test_bought_count = 0
    evalue = np.zeros(10)
    right_user = np.zeros(100)
    result = []
    for i in range(u_num):
        if (i+1) % 1000 == 0:
            print(i+1)
        test_score = []
        one_test_data = test_data[i]
        test_bought_count += len(one_test_data) - 1000
        for j in one_test_data:
            rij = mud.get_rating(i,j)
            aij = mud.get_aij(i,j)
            sigij = 2 /(1+math.exp(-rij)) - 1
            sigpij = 1/(1+math.exp(-item_price[j]))
            sij = aij * sigij /sigpij
            test_score.append(sij)
        sorted_test = np.argsort(test_score).tolist()
        for k in range(1,11,1):
            b = budget[i] * 1.0
            m = 1
            sorted_test_1 = []
            while(len(sorted_test_1) <= k):
                if m >= len(one_test_data):
                    b += budget[i] * 0.1
                    m =1
                one_test = sorted_test[-m]
                if item_price[one_test_data[one_test]] <= b:
                    evalue[k-1] += test_score[one_test]
                    sorted_test_1.append(one_test)
                m += 1
            for r in sorted_test_1:
                if r >= 1000:
                    right_num[k-1] += 1
            if k == 10:
                result.append(sorted_test_1)
        for k in range(1,101,1):
            b = budget[i] * 1.0
            m = 1
            sorted_test_1 = []
            while(len(sorted_test_1) <= k):
                if m >= len(one_test_data):
                    b += budget[i] * 0.1
                    m =1
                one_test = sorted_test[-m]
                if item_price[one_test_data[one_test]] <= b:
                    sorted_test_1.append(one_test)
                m += 1
            for r in sorted_test_1:
                if r >= 1000:
                    right_user[k-1] += 1
                    break
    right_user = right_user/u_num
    for i in range(10):
        print("================")
        print("top " + str(i+1))
        print("precision:")
        pre = right_num[i]/(u_num*(i+1))
        print(pre)
        print("recall:")
        recall = right_num[i]/test_bought_count
        print(recall)
        print("F1-measure:")
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
        print(f1)
        print("evalue:")
        print(evalue[i]/u_num)
    
def main(category,u_num,i_num):
    params = dict()
    params['CFK'] = 50
    params['CFlr'] = 1e-4
    params['CFbeta'] = 1.5
    params['CFiterations'] = 3
    params['K'] = 100
    params['lr'] = 5e-4
    params['beta'] = 1
    params['iterations'] = 3
    TrainTest(category,u_num,i_num,params)
        
if __name__ == "__main__":
    main("Baby",23894,39767)
    
    