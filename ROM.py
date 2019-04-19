import numpy as np
import torch
import torch.nn as nn

class ROM(nn.Module):
	"""docstring for ROM"""
	def __init__(self, userLen, itemLen, params):
		super(ROM, self).__init__()
		self.userNum = userLen
		self.itemNum = itemLen
		self.params = params
		if 'gpu' in params and params['gpu'] == True:
			self.device = 'cuda'
		else:
			self.device = 'cpu'
		l_size = params['l_size']
		self.uEmbed = nn.Embedding(userLen, l_size)
