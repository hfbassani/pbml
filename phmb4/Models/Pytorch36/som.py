# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch as torch
import numpy as np
import torch.nn as nn
import pandas as pd
import re
from itertools import cycle

import gc
from memory_profiler import profile
import torch.cuda as cutorch

import subprocess
from larfdssom import LARFDSSOM


# fxn taken from https://discuss.pytorch.org/t/memory-leaks-in-trans-conv/12492
def get_gpu_memory_map():   
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    
    return float(result)

class SSSOM(LARFDSSOM):

	def __init__(self, use_cuda, ngpu, dim, max_node_number=70, no_class=999,
				 a_t=0.3, lp=0.001, dsbeta=0.0001, age_wins=100, e_b=0.5,
				 e_n=0.01, eps_ds=0.01, minwd=0.5, epochs=100, e_push=0.001):

		super(SSSOM, self).__init__(use_cuda, ngpu, dim, max_node_number,
									a_t, lp, dsbeta, age_wins,
									e_b, e_n, eps_ds, minwd, epochs)

		self.no_class = torch.tensor(no_class, device=self.device)

		self.e_push = torch.tensor(e_push, device=self.device) * self.e_b

		# Initialize the map Map
		self.classes = None

	def forward(self, x, y=None):
		x = x.to(self.device)
		y = y.to(self.device)

		batch_size = x.size(0)


		print("####################################################################")
		#print(x.size())
		#print(y.size())
		#print(batch_size)
		## VARIAVEIS EM GPU ...
		#print(self.a_t.size())
		#print(self.lp.size())
		#print(self.dsbeta.size())
		#print(self.age_wins.size())
		#print(self.e_b.size())
		#print(self.e_n.size())
		#print(self.eps_ds.size())
		#print(self.minwd.size())
		#print(self.epochs.size())
		


		#import time
		#time.sleep(2)
		

		calculate_mean = True if batch_size > 1 else False

		if self.weights is None:
			#print(self.weights.size())
			self.initialize_map(x, y, calculate_mean)
			#print(self.weights.size())
			#exit(0)

		elif batch_size > 1:
			x_sup, y_sup = x[y != self.no_class], y[y != self.no_class]
			if x_sup.size(0) > 0:
				print(x_sup.size(), y_sup.size())
				self.update_map_sup(x_sup, y_sup)

			x_unsup = x[y == self.no_class]
			if x_unsup.size(0) > 0:
				self.update_map(x_unsup)

		else:
			if torch.eq(y, self.no_class):
				self.update_map(x)
			else:
				self.update_map_sup(x, y)

		if self.step >= self.age_wins:
			self.remove_loosers()
			self.wins = torch.zeros(self.wins.size(0), device=self.device)
			self.update_all_connections()

			self.step = 0

		self.step += 1
		#print("Weights Size: ", self.weights.size())
		#print("Moving Average Size: ", self.moving_avg.size())
		#print("Relevances Size:", self.relevances.size())
		#print("Neighbors Size: ", self.neighbors.size())
		#print("Wins Size: ", self.wins.size())
		#print("####################################################################")
	
	#JA ANALISADO #@profile
	def update_map_sup(self, w, y):
		batch_size = w.size(0)

		if batch_size > 1:
			self.batch_update_map_sup(w, y)
		else:
			activations = self.activation(w)
			ind_max = torch.argmax(activations)
			ind_max = torch.full((1,), ind_max, dtype=torch.int64, device=self.device)

			if self.classes[ind_max] == y or self.classes[ind_max] == self.no_class:
				if activations[ind_max] < self.a_t and self.weights.size(0) < self.max_node_number:
					self.add_node(w, y)

				elif activations[ind_max] >= self.a_t:
					self.wins[ind_max] += 1
					self.classes[ind_max] = y
					self.update_all_connections()
					# self.update_connections(ind_max)
					self.update_node(w, self.e_b, ind_max)

					self.update_neighbors(w, ind_max)
			else:
				self.handle_different_class(activations, w, y)

	#JA ANALISADO #@profile
	def batch_update_map_sup(self, w, y):
		activations = self.activation(w)
		activations_max, indices_max = torch.max(activations, dim=1)

		# separate samples according to the winner node
		for grouped_w, grouped_y, node in self.group_by_winner_node(indices_max, w, y):
			local_activations = activations[indices_max == node]
			local_max_activation = activations_max[indices_max == node]
			same_classes = self.classes[node] == grouped_y
			win_no_class = self.classes[node] == self.no_class
			class_criterion = same_classes | win_no_class

			class_criterion_nodes = class_criterion.nonzero()
			if class_criterion_nodes.size(0) > 0:
				class_criterion_nodes = class_criterion_nodes.squeeze(1)

				geq_at = (local_max_activation[class_criterion_nodes] >= self.a_t).nonzero()
				lt_at = (local_max_activation[class_criterion_nodes] < self.a_t).nonzero()
				indices_geq_at = None if geq_at.size(0) == 0 else geq_at.squeeze(1)
				indices_lt_at = None if lt_at.size(0) == 0 else lt_at.squeeze(1)

				if indices_lt_at is not None and self.weights.size(0) + indices_lt_at.size(0) < self.max_node_number:
					add_w, add_y = self.group_data_by_mean(grouped_w[indices_lt_at], grouped_y[indices_lt_at])
					self.add_node(add_w, add_y)

				if indices_geq_at is not None:  # duas amostras de rotulos diferentes ganharam pro mesmo nodo (que pode ter um rotulo, ou nao), o que fazer?
					new_w, new_y = self.group_data_by_mean(grouped_w[indices_geq_at], grouped_y[indices_geq_at])

					if new_w.size(0) == 1:
						self.wins[node] += 1
						self.classes[node] = new_y
						self.update_all_connections()
						self.update_node(new_w, self.e_b, node)

						self.update_neighbors(new_w, node)
					else:
						instances = new_w.size(0) - 1
						new_ind = self.clone_node(node, instances)

						self.wins[new_ind] += 1
						self.classes[new_ind] = new_y
						self.update_all_connections()
						self.update_node(new_w, self.e_b, new_ind)

						self.update_neighbors(new_w, new_ind)

			different_class_nodes = (class_criterion == 0).nonzero()
			if different_class_nodes.size(0) > 0:
				different_class_nodes = different_class_nodes.squeeze(1)

				self.handle_different_class(local_activations[different_class_nodes],
											grouped_w[different_class_nodes],
											grouped_y[different_class_nodes])

	#JA ANALISADO #@profile #NAO APARECEU NO LOG
	def clone_node(self, node, instances):
		indices = torch.range(0, instances - 1, device=self.device, dtype=torch.long) + self.weights.size(0)

		weights = torch.stack([self.weights[node]] * instances)
		cls = self.classes[node]

		self.add_node(weights, torch.stack([cls] * instances))

		self.moving_avg[indices] = torch.tensor(self.moving_avg[node])
		self.relevances[indices] = torch.tensor(self.relevances[node])
		self.neighbors[indices] = torch.tensor(self.neighbors[node])
		self.wins[indices] = torch.tensor(self.wins[node])

		return torch.cat((indices, node.unsqueeze(0))).sort()[0]
	
	#JA ANALISADO #@profile
	def handle_different_class(self, activations, w, y):
		new_winners, wrong_winners = self.next_winners(activations, y)
		if wrong_winners is None and new_winners is None:
			new_w, new_y = self.group_data_by_mean(w, y)
			self.add_node(new_w, new_y)

		else:
			no_new_winners = (new_winners == -1).nonzero()
			if no_new_winners.size(0) > 0:
				no_new_winners = no_new_winners.squeeze(1)
				new_w, new_y = self.group_data_by_mean(w[no_new_winners], y[no_new_winners])
				self.add_node(new_w, new_y)

			valid_new_winners = (new_winners != -1).nonzero()
			if valid_new_winners.size(0) > 0:
				valid_new_winners = valid_new_winners.squeeze(1)

				curr_winners = new_winners[valid_new_winners]
				curr_wrong_winners = wrong_winners[valid_new_winners]

				for new_w, new_y, new_winner, wrong_winner in zip(w[valid_new_winners], y[valid_new_winners], curr_winners, curr_wrong_winners):
					self.update_node(new_w, -self.e_push, wrong_winner)
					self.wins[new_winner] += 1

					self.update_node(new_w, self.e_b, new_winner)

					self.update_neighbors(new_w, new_winner)

	#JA ANALISADO #@profile #NAO APARECEU NO LOG
	def next_winners(self, activations, y):
		wrong_winners = None
		winners = None

		sorted_activations, indexes = torch.sort(activations, descending=True)

		if len(indexes.size()) == 1:  # len(index.size()) == 1 means that it is not a batch operation
			indexes = indexes[sorted_activations >= self.a_t]
			winners, wrong_winners = self.get_next(indexes, indexes, winners, wrong_winners, y)

		else:
			for local_act, local_ind, local_y in zip(sorted_activations, indexes, y):
				geq_indexes = local_ind[local_act >= self.a_t]
				winners, wrong_winners = self.get_next(local_ind, geq_indexes, winners, wrong_winners, local_y)

		return winners, wrong_winners

	#JA ANALISADO #@profile #NAO APARECEU NO LOG
	def get_next(self, curr, indexes, winners, wrong_winners, y):
		if curr.size(0) > 0 and indexes.size(0) > 0:
			wrong = torch.full((1,), curr[0], dtype=torch.int64, device=self.device)

			if wrong_winners is None:
				wrong_winners = wrong
			else:
				wrong_winners = torch.cat((wrong_winners, wrong))

			no_classes = self.classes[indexes] == self.no_class
			same_classes = self.classes[indexes] == y
			curr_winner = indexes[no_classes | same_classes]

			curr_winner_size = curr_winner.size(0)
			if curr_winner_size == 0:
				curr_winner = torch.full((1,), -1, dtype=torch.int64, device=self.device)
			else:
				curr_winner = torch.full((1,), curr_winner[0], dtype=torch.int64, device=self.device)

			if winners is None:
				winners = curr_winner
			else:
				winners = torch.cat((winners, curr_winner))
		else:
			invalid = torch.full((1,), -1, dtype=torch.int64, device=self.device)

			if wrong_winners is None:
				wrong_winners = invalid
			else:
				wrong_winners = torch.cat((wrong_winners, invalid))

			if winners is None:
				winners = invalid
			else:
				winners = torch.cat((winners, invalid))

		return winners, wrong_winners

	#JA ANALISADO #@profile #NAO APARECEU NO LOG
	def initialize_map(self, w, y=None, calculate_mean=False):

		if not calculate_mean:
			super(SSSOM, self).initialize_map(w, y, calculate_mean)

			if y is None:
				y = self.no_class

			self.classes = y.to(self.device)
		else:

			if y is None:
				y = torch.full((w.size(0),), self.no_class, dtype=self.no_class.dtype, device=self.device)

			new_w, new_y = self.group_data_by_mean(w, y)
			super(SSSOM, self).initialize_map(new_w, new_y, calculate_mean=False)
			self.classes = new_y.to(self.device)

	#JA ANALISADO #@profile #NAO APARECEU NO LOG
	def add_node(self, w, y=None):
		if y is None:
			y = torch.full((w.size(0),), self.no_class, dtype=self.no_class.dtype, device=self.device)

		self.classes = torch.cat((self.classes, y.to(self.device)))
		super(SSSOM, self).add_node(w, y)

	#JA ANALISADO #@profile #NAO APARECEU NO LOG
	'''
	def update_connections(self, node_index):
		dists = self.relevance_distances(self.relevances[node_index], self.relevances)
		print(dists.size())
		dists_connections = (dists < self.minwd)

		stacked_classes = torch.stack([self.classes] * node_index.size(0))
		stacked_classes_transposed = stacked_classes.t()
		classes_connections = (stacked_classes == self.no_class) | (stacked_classes_transposed == self.no_class) | (stacked_classes.t() == self.classes[node_index]).t()
		connections = dists_connections & classes_connections

		for node, connection in zip(node_index, connections):
			self.neighbors[node] = connection
			self.neighbors[node][node] = 0
	'''

	#JA ANALISADO #@profile #NAO APARECEU NO LOG
	def remove_loosers(self):
		remaining_idxs = (self.wins >= self.step * self.lp).nonzero()

		if remaining_idxs.size(0) > 0:
			self.classes = self.classes[remaining_idxs.squeeze(1)]

			super(SSSOM, self).remove_loosers(remaining_idxs)

	def update_all_connections(self):
		# if self.relevances.size(0) > 1:
		dists = self.relevance_distances(self.relevances, self.relevances)
		dists_connections = dists < self.minwd

		stacked_classes = torch.stack([self.classes] * self.classes.size(0))
		stacked_classes_transposed = stacked_classes.t()
		classes_connections = (stacked_classes == self.no_class) | (stacked_classes_transposed == self.no_class) | (stacked_classes == stacked_classes_transposed)

		connections = dists_connections & classes_connections

		self.neighbors = (1 - torch.eye(self.weights.size(0), dtype=torch.uint8, device=self.device)) * connections

	def cluster_classify(self, dataloader, is_subspace, filter_noise):
		clustering_classify = pd.DataFrame(columns=['sample_ind', 'cluster', 'class'])
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			activations = self.activation(inputs.to(self.device))
			ind_max = torch.argmax(activations)

			if filter_noise and activations[ind_max] < self.a_t:
				continue

			if not is_subspace:

				if self.classes[ind_max] == self.no_class:
					_, indices = torch.sort(activations, descending=True)

					classes = self.classes[indices] != self.no_class

					winners = indices[classes]

					if winners.size(0) > 0:
						ind_max = winners[0]

				clustering_classify = clustering_classify.append({'sample_ind': batch_idx,
																  'cluster': ind_max.item(),
																  'class': self.classes[ind_max].item()},
																 ignore_index=True)
			else:
				x = 1

		return clustering_classify