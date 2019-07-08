import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

DEBUG = False


def sparsity(interactions):
	print("Calculating sparsity...")
	users = np.unique(interactions[:,0])
	items = np.unique(interactions[:,1])
	N = 0
	for u in tqdm(users):
		uniqueUserInteractions = np.unique(interactions[interactions[:,0] == u, 1])
		N += len(uniqueUserInteractions)
	return float(N) / (len(users) * len(items))

def rmse(predictions, targets, cmp_max = False):
	assert predictions.shape == targets.shape
	if cmp_max:
		predictions = [np.argmax(r)+1 for r in predictions]
		targets = [np.argmax(r)+1 for r in targets]
	error = predictions - targets
	return np.sqrt(np.mean(error ** 2))

# confusion matrix
# accuracy, precision, recall, f1-score, rmse
def stat_analyze(predictions, targets):
	assert predictions.shape == targets.shape
	from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
	results = dict()
	if DEBUG:
		print(predictions.shape)
		print(predictions[:10])
		print(np.unique(predictions))
		print(targets.shape)
		print(targets[:10])
		print(np.unique(targets))
		input()
	# confusion matrix
	cf = confusion_matrix(predictions, targets)
	print("\tConfusion matrix")
	print(cf)
	results['confusion matrix'] = cf
	# accuracy
	accuracy = accuracy_score(targets, predictions, normalize = True)
	print("\n\tAccuracy: " + str(accuracy))
	results['accuracy'] = accuracy
	# precision
	precision = precision_score(targets, predictions, average = None)
	precision_average = np.mean(precision)
	precision_weighted = precision_score(targets, predictions, average = 'weighted')
	print("\tPrecision: " + str(precision))
	print("\tAverage precision: " + str(precision_average))
	print("\tWeighted precision: " + str(precision_weighted))
	results['precision'] = precision
	results['precision_average'] = precision_average
	results['precision_weighted'] = precision_weighted
	# recall
	recall = recall_score(targets, predictions, average = None)
	recall_average = np.mean(recall)
	recall_weighted = recall_score(targets, predictions, average = 'weighted')
	print("\tRecall: " + str(recall))
	print("\tAverage recall: " + str(recall_average))
	print("\tWeighted recall: " + str(recall_weighted))
	results['recall'] = recall
	results['recall_average'] = recall_average
	results['recall_weighted'] = recall_weighted
	# f1 score
	f1 = f1_score(targets, predictions, average = None)
	f1_average = np.mean(f1)
	f1_weighted = f1_score(targets, predictions, average = 'weighted')
	print("\tF1-score: " + str(f1))
	print("\tAverage f1-score: " + str(f1_average))
	print("\tWeighted f1-score: " + str(f1_weighted))
	results['f1'] = f1
	results['f1_average'] = f1_average
	results['f1_weighted'] = f1_weighted
	#print("\n\tSample predictions and actual values")
	#print(predictions[:20])
	#print(targets[:20])
	# RMSE
	error = rmse(predictions, targets)
	print("\tRMSE: " + str(error))
	results['rmse'] = error

	# return results
	return results

def user_analyze(user_ids, item_ids, predictions, targets):
	assert user_ids.shape[0] == item_ids.shape[0]
	assert item_ids.shape[0] == predictions.shape[0]
	assert predictions.shape[0] == targets.shape[0]
	from sklearn.metrics import confusion_matrix

	users = np.unique(user_ids)
	userRatingCube = dict()
	uniqueTargets = np.unique(targets)
	print("\tunique targets: " + str(uniqueTargets))
	print("\ttargets: " + str(targets.shape) + " - " + str(targets[:20]))
	print("\tsample predictions: " + str(predictions[:20]))
	for u in users:
		ids = (user_ids == u)
		userRatingCube[u] = confusion_matrix(predictions[ids], targets[ids], labels = uniqueTargets)
	return userRatingCube

def analyse_user_rating_cube(fPath):
	#fPath = "results/test1/user_rating_cube_um2_cs3_v16_lr0.003_epoch20_struct[16, 32, 1]"
	#fPath = "results/test1/user_rating_cube_um2_cs3_v64_lr0.003_epoch20_struct[64, 1]"
	#fPath = "results/test1/user_rating_cube_um2_cs3_v32_lr0.005_epoch20_struct[32, 32, 1]"
	userRatingCube = pickle.load(open(fPath, "rb"))
	print("Load file: " + fPath)
	userList = list()
	userHisSizeList = list()
	nUser5Less = 0
	accList = list()
	for u, cf in userRatingCube.items():
		cf = np.array(cf)
		userList.append(u)
		nRecord = np.sum(cf)
		userHisSizeList.append(nRecord)
		if nRecord <= 10:
			nUser5Less = nUser5Less + 1
		acc = float(cf[0,0]+cf[1,1]+cf[2,2]+cf[3,3]+cf[4,4]) / nRecord
		accList.append(acc)
	accList = np.array(accList)
	userHisSizeList = np.array(userHisSizeList)
	# user history length: power law
	print("User with 10 or less records: {0:.2%}".format(float(nUser5Less) / len(userList)))
	plt.hist(userHisSizeList, bins = 200)
	plt.ylabel("number of users")
	plt.xlabel("history length")
	plt.show()
	# user prediction accuracy: normal distribution
	print("Lowest accuracy: " + str(np.min(accList)))
	plt.hist(accList, bins = 200)
	plt.ylabel("number of users")
	plt.xlabel("accuracy")
	plt.show()
	# user history length influence accuracy
	plt.hist(accList[userHisSizeList < 100], bins = 100)
	plt.title("When user history length < 100")
	plt.ylabel("number of users")
	plt.xlabel("accuracy")
	plt.show()
	plt.hist(accList[userHisSizeList > 100], bins = 100)
	plt.title("When user history length > 100")
	plt.ylabel("number of users")
	plt.xlabel("accuracy")
	plt.show()

def rating_analysis(score_dict):
	"""
	Input:
	- score_dict: dictionary of (user: (scores, mse)); 
		- where scores is: an array [pred_score]
	"""
	SSE = 0.
	count = 0
	minR = 5.
	maxR = 0.
	for u, (scores, mse) in score_dict.items():
		SSE = SSE + mse * len(scores)
		count = count + len(scores)
		u, v = torch.min(scores), torch.max(scores)
		if u < minR:
			minR = u
		if v > maxR:
			maxR = v 
	print("\tresponse range: [" + str(minR) + ", " + str(maxR) + "]")
	return {"rmse": np.sqrt(SSE / count), "min": minR, "max": maxR}

def ranking_analysis(score_dict, k):
	"""
	Input:
	- score_dict: dictionary of (user: (scores, #ground_truth)); 
			- where scores is: a heap array of (pred_score, (itemId, is_ground_truth))
	"""
	assert k > 0
	avgPrec = np.zeros(k)
	avgRecall = np.zeros(k)
	avgNDCG = np.zeros(k)
	for u, (itemScores, n_gt) in score_dict.items():
		if n_gt <= 0:
			continue
		userPrecision = np.zeros(k)
		userRecall = np.zeros(k)
		userDCG = np.zeros(k)
		userIDCG = np.zeros(k)
		hit = 0
		N = 0
		scores = []
		for score, (itemId, rel) in itemScores:
			scores.append(score)
			if rel == 1:
				hit = hit + 1
			N = N + 1
			if N > k:
				break
			# recall
			userRecall[N-1] = float(hit) / n_gt
			# precision
			userPrecision[N-1] = float(hit) / N
			# ndcg
			if N == 1:
				userDCG[0] = float(rel)
				userIDCG[0] = 1.0
			else:
				userDCG[N-1] = userDCG[N-2] + float(rel) / np.log2(N)
				if N <= n_gt:
					userIDCG[N-1] = userIDCG[N-2] + 1.0 / np.log2(N)
				else:
					userIDCG[N-1] = userIDCG[N-2]
			if DEBUG:
				print("{score: " + str(score) + "; rel: " + str(rel) + "}")
		#input()
		avgPrec += userPrecision
		avgRecall += userRecall
		avgNDCG += (userDCG / userIDCG)
	avgPrec /= len(score_dict)
	avgRecall /= len(score_dict)
	avgNDCG /= len(score_dict)
	print("\tPrecision@: {1:" + str(avgPrec[0]) + "; 5: " + str(avgPrec[4]) + "; 10: " + str(avgPrec[9]) + "}")
	print("\tRecall@: {1:" + str(avgRecall[0]) + "; 5: " + str(avgRecall[4]) + "; 10: " + str(avgRecall[9]) + "}")
	print("\tNDCG@: {1:" + str(avgNDCG[0]) + "; 5: " + str(avgNDCG[4]) + "; 10: " + str(avgNDCG[9]) + "}")
	return {"avg_precision": avgPrec, "avg_recall": avgRecall, "avg_ndcg": avgNDCG}

###########################################################################################
# Observations
def user_history_distribution(interaction):
	users = np.unique(interaction[:,0])
	#userHist = dict()
	hLenList = np.zeros_like(users)
	for i in tqdm(range(len(users))):
		hLenList[i] = len(interaction[interaction[:,0] == users[i]])
	hLenList.sort()
	logList = np.log(hLenList)
	plt.plot(logList[::-1])
	plt.show()
	plt.close()
	for l in logList:
		print(l)

def user_response_variance_distribution(interaction):
	users = np.unique(interaction[:,0])
	varList = np.zeros_like(users)
	for i in tqdm(range(len(users))):
		varList[i] = np.var(interaction[interaction[:,0] == users[i], 2])
	varList.sort()
	plt.plot(varList[::-1])
	plt.show()
	plt.close()
	for v in varList:
		print(v)

def user_variance_vs_performance(score_dict, user_hist):
	users = np.unique(dataset[:,0])
	histLen = list()
	respVar = list()
	perfScores = np.zeros((len(users), k))
	k = 50
	for i in range(len(users)):
		u = users[i]
		hist = user_hist[u]
		# history length
		histLen.append(len(hist))
		# response variance
		respVar.append(np.var(hist[:, 1]))
		# NDCG score
		(itemScores, nNeg) = scores[u]
		N = 0
		scores = []
		userDCG = np.zeros(k)
		userIDCG = np.zeros(k)
		for score, (itemId, rel) in itemScores:
			scores.append(score)
			N = N + 1
			if N > k:
				break
			# ndcg
			if N == 1:
				userDCG[0] = float(rel)
				userIDCG[0] = 1.0
			else:
				userDCG[N-1] = userDCG[N-2] + float(rel) / np.log2(N)
				if N <= nNeg:
					userIDCG[N-1] = userIDCG[N-2] + 1.0 / np.log2(N)
				else:
					userIDCG[N-1] = userIDCG[N-2]
		avgNDCG =(userDCG / userIDCG)
		perfScore[i,:] = avgNDCG
	plt.scatter(perfScore[:,0], histLen)
	plt.show()
	plt.close()
	return histLen, respVar, perfScore

def plot_main():
	"""
	See drive for the full data
	"""
	fig = plt.figure()
	rmseList = np.array([\
		[[0.929218703, 0.921731935], [0.932686257, 0.915460113]],\
		[[0.869248225, 0.856207401], [0.872315194, 0.852395141]]])
	datasetNames = ["MovieLens 100K", "MovieLens1M"]
	index = np.arange(2)
	barWidth = 0.35
	opacity = 0.4
	for i in range(2):
		data = rmseList[i]
		ax = plt.subplot(1,2,i+1)
		b1 = ax.bar(index, data[0,:], barWidth, alpha = opacity, color = 'b')
		b2 = ax.bar(index+barWidth, data[1,:], barWidth, alpha = opacity, color = 'g')
		ax.set_title(datasetNames[i])
		ax.set_xlabel('Model')
		ax.set_ylabel('RMSE')
		ax.set_xticks(index + barWidth / 2)
		ax.set_xticklabels(('MF', 'NeuMF'))
	fig.legend((b1, b2), ('SUMO', 'PL-DUMO'), 'upper center', ncol = 2)
	plt.show()

if __name__ == "__main__":
	#import data_extract
	#train, val, test = data_extract.read_movielens_100k()
	#train, val, test = data_extract.read_movielens_1m()
	#user_history_distribution(train)
	#user_response_variance_distribution(train)
	plot_main()