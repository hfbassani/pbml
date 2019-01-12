# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch as torch
import numpy as np
import torch.nn as nn
import pandas as pd
import re


class LARFDSSOM(nn.Module):

    def __init__(self, use_cuda, ngpu, dim, max_node_number=70,
                 a_t=0.3, lp=0.001, dsbeta=0.0001, age_wins=100,
                 e_b=0.5, e_n=0.01, eps_ds=0.01, minwd=0.5, epochs=100):
        super(LARFDSSOM, self).__init__()

        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.ngpu = ngpu
        self.step = 0
        self.dim = dim
        self.max_node_number = max_node_number
        self.a_t = torch.tensor(a_t, device=self.device)
        self.lp = torch.tensor(lp, device=self.device)
        self.dsbeta = torch.tensor(dsbeta, device=self.device)
        self.age_wins = torch.tensor(age_wins, device=self.device)
        self.e_b = torch.tensor(e_b, device=self.device)
        self.e_n = torch.tensor(e_n, device=self.device) * self.e_b
        self.eps_ds = torch.tensor(eps_ds, device=self.device)
        self.minwd = torch.tensor(minwd, device=self.device)
        self.epochs = torch.tensor(epochs, device=self.device)

        # Initialize the map Map
        self.weights = None
        self.moving_avg = None
        self.relevances = None
        self.neighbors = None
        self.wins = None

    def forward(self, x, y=None):
        x = x.to(self.device)
        y = y.to(self.device)

        batch_size = x.size(0)
        calculate_mean = True if batch_size > 1 else False

        if self.weights is None:
            self.initialize_map(x, y, calculate_mean)
        else:
            self.update_map(x)

        if self.step >= self.age_wins:
            self.remove_loosers()
            self.wins = torch.zeros(self.wins.size(0), device=self.device)
            self.update_all_connections()

            self.step = 0

        self.step += 1

    def initialize_map(self, w, y=None, calculate_mean=False):

        batch_size = w.size(0)

        if not calculate_mean:
            self.weights = torch.tensor(w, device=self.device)
        else:
            w_new = self.group_data_by_mean(w, y)
            batch_size = w_new.size(0)
            self.weights = w_new.to(self.device)

        self.moving_avg = torch.zeros(batch_size, self.dim, device=self.device)
        self.relevances = torch.ones(batch_size, self.dim, device=self.device)

        self.neighbors = torch.zeros(batch_size, batch_size, dtype=torch.uint8, device=self.device)
        self.wins = torch.zeros(batch_size, device=self.device)

    def group_data_by_mean(self, w, y=None):
        new_w = None
        unique_targets = None

        if y is None:
            new_w = w.mean(dim=0).unsqueeze(0)
        else:

            unique_targets = torch.from_numpy(np.unique(y)).to(
                self.device)  # there is no GPU support for unique operation
            for target in unique_targets:
                target_occurrences = (y == target).nonzero().squeeze(1)
                w_target = w[target_occurrences].mean(dim=0).unsqueeze(0)

                if new_w is None:
                    new_w = w_target
                else:
                    new_w = torch.cat((new_w, w_target), 0)

        return new_w, unique_targets

    def update_map(self, w):
        batch_size = w.size(0)

        if batch_size > 1:
            self.batch_update_map(w)
        else:
            activations = self.activation(w)[0]
            ind_max = torch.argmax(activations)
            if activations[ind_max] < self.a_t and self.weights.size(0) < self.max_node_number:
                self.add_node(w)
            elif activations[ind_max] >= self.a_t:
                self.wins[ind_max] += 1
                self.update_node(w, self.e_b, ind_max)

                self.update_neighbors(w, ind_max)

    def batch_update_map(self, w):
        activations = self.activation(w)
        activations_max, indices_max = torch.max(activations, dim=1)

        geq_at = (activations_max >= self.a_t).nonzero()
        lt_at = (activations_max < self.a_t).nonzero()
        indices_geq_at = None if geq_at.size(0) == 0 else geq_at.squeeze(1)
        indices_lt_at = None if lt_at.size(0) == 0 else lt_at.squeeze(1)
        if indices_lt_at is not None and self.weights.size(0) + indices_lt_at.size(0) < self.max_node_number:
            self.add_node(w[indices_lt_at])

        if indices_geq_at is not None:
            for grouped_w, _, node in self.group_by_winner_node(indices_max[indices_geq_at], w[indices_geq_at]):
                new_w, _ = self.group_data_by_mean(grouped_w)

                self.wins[node] += 1
                self.update_all_connections()
                self.update_node(new_w, self.e_b, node)

                self.update_neighbors(new_w, node)

    def group_by_winner_node(self, winners, w, y=None):
        groups = None

        unique_nodes = torch.from_numpy(np.unique(winners)).to(
            self.device)  # there is no GPU support for unique operation

        for node in unique_nodes:
            node_occurrences = (winners == node).nonzero().squeeze(1)
            w_target = w[node_occurrences]
            y_target = None if y is None else y[node_occurrences]

            if groups is None:
                groups = [(w_target, y_target, node)]
            else:
                groups.append((w_target, y_target, node))

        return groups

    def update_neighbors(self, w, nodes):
        if len(nodes.size()) == 0:  # len(nodes.size()) == 0 means that it is a scalar tensor
            self.update_node_neighbors(nodes, w)

        else:
            for node, w_update in zip(nodes, w):
                self.update_node_neighbors(node, w_update)

    def update_node_neighbors(self, node, w):
        # check if the node has any neighbors
        has_neighbors = self.neighbors[node].sum().nonzero().squeeze(0)

        if has_neighbors.size(0) > 0:
            # get neighbors
            node_neighbors = self.neighbors[node].nonzero().squeeze(1)
            self.update_node(w, self.e_n, node_neighbors)

    def activation(self, w):
        dists = self.weighted_distance(w)
        relevances_sum = torch.sum(self.relevances, 1)

        return torch.div(relevances_sum, torch.add(torch.add(relevances_sum, dists), 1e-7))

    def add_node(self, w, y=None):
        batch_size = w.size(0)

        self.weights = torch.cat((self.weights, w), 0)

        zeros = torch.zeros(batch_size, self.dim, device=self.device)
        ones = torch.ones(batch_size, self.dim, device=self.device)

        self.moving_avg = torch.cat((self.moving_avg, zeros), 0)
        self.relevances = torch.cat((self.relevances, ones), 0)

        self.neighbors = torch.cat((self.neighbors,
                                    torch.zeros(self.neighbors.size(0), batch_size, dtype=torch.uint8,
                                                device=self.device)), 1)
        self.neighbors = torch.cat((self.neighbors,
                                    torch.zeros(batch_size, self.neighbors.size(1), dtype=torch.uint8,
                                                device=self.device)), 0)

        # update_ind = torch.range(0, batch_size - 1, device=self.device, dtype=torch.long) + self.weights.size(0) - 1
        # self.update_connections(update_ind)
        self.update_all_connections()

        self.wins = torch.cat((self.wins, torch.zeros(batch_size, device=self.device)))

    def update_connections(self, node_index):
        dists = self.relevance_distances(self.relevances[node_index], self.relevances)
        dists_connections = (dists < self.minwd)
        dists_connections[node_index] = 0

        self.neighbors[node_index] = dists_connections

    def update_node(self, w, lr, index):
        distance = torch.abs(torch.sub(w, self.weights[index]))
        self.moving_avg[index] = torch.mul(lr * self.dsbeta, distance) + torch.mul(1 - lr * self.dsbeta,
                                                                                   self.moving_avg[index])

        if len(index.size()) == 0:  # len(index.size()) == 0 means that it is a scalar tensor
            maximum = torch.max(self.moving_avg[index])
            minimum = torch.min(self.moving_avg[index])
            avg = torch.mean(self.moving_avg[index])
        else:
            maximum = torch.max(self.moving_avg[index], 1)[0].unsqueeze(1)
            minimum = torch.min(self.moving_avg[index], 1)[0].unsqueeze(1)
            avg = torch.mean(self.moving_avg[index], 1).unsqueeze(1)

        one_tensor = torch.tensor(1, dtype=torch.float, device=self.device)

        self.relevances[index] = torch.div(one_tensor,
                                           one_tensor + torch.exp(torch.div(torch.sub(self.moving_avg[index], avg),
                                                                            torch.mul(self.eps_ds,
                                                                                      torch.sub(maximum, minimum)))))
        self.relevances[self.relevances != self.relevances] = 1.  # if (max - min) == 0 then set to 1

        self.weights[index] = torch.add(self.weights[index], torch.mul(lr, torch.sub(w, self.weights[index])))

    def remove_loosers(self, remaining_indexes=None):

        remaining_idxs = remaining_indexes if remaining_indexes is not None else (
                self.wins >= self.step * self.lp).nonzero()

        if remaining_idxs.size(0) > 0:
            remaining_idxs = remaining_idxs.squeeze(1)

            self.weights = self.weights[remaining_idxs]
            self.moving_avg = self.moving_avg[remaining_idxs]
            self.relevances = self.relevances[remaining_idxs]

            self.update_all_connections()

            self.wins = torch.zeros(remaining_indexes.size(0), device=self.device)

    def update_all_connections(self):
        dists_stacked = torch.stack([self.relevances] * self.relevances.size(0))
        dists_stacked_transposed = dists_stacked.t()
        dists = self.relevance_distances(dists_stacked, dists_stacked_transposed)
        dists_connections = dists < self.minwd

        self.neighbors = (1 - torch.eye(self.weights.size(0), dtype=torch.uint8,
                                        device=self.device)) * dists_connections

    def weighted_distance(self, w):
        dists = self.pairwise_distances(w, self.weights)

        relevances_new = self.relevances.sum(1).view(1, -1)

        dist_weight = dists * (relevances_new / self.relevances.size()[-1])

        dist_weight[dist_weight != dist_weight] = 0

        return dist_weight

    def relevance_distances(self, x1, x2):
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)

        return self.pairwise_distances(x1, x2)

    def pairwise_distances(self, x1, x2=None):

        x1_norm = (x1 ** 2).sum(1).view(-1, 1)

        x2_t = torch.transpose(x2, 0, 1)
        x2_norm = (x2 ** 2).sum(1).view(1, -1)

        dist = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2_t)

        dist[dist != dist] = 0  # replace nan values with 0

        return dist

    def organization(self, dataloader):
        for epoch in range(self.epochs):
            print("Epoch: ", epoch, " of ", self.epochs)
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                self.forward(inputs, targets)
        self.convergence(dataloader)

    def convergence(self, dataloader):
        curr_step = self.step
        total_steps = 2 * self.age_wins
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if self.step == 1:
                self.max_node_number = self.weights.size(0)
            if curr_step >= total_steps:
                break

            self.forward(inputs, targets)
            curr_step += 1

    def cluster(self, dataloader, is_subspace, filter_noise):
        clustering = pd.DataFrame(columns=['sample_ind', 'cluster'])
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            activations = self.activation(inputs.to(self.device))
            ind_max = torch.argmax(activations).item()

            if filter_noise and activations[ind_max] < self.a_t:
                continue

            if not is_subspace:
                clustering = clustering.append({'sample_ind': batch_idx, 'cluster': ind_max}, ignore_index=True)
            else:
                x = 1

        return clustering

    def write_output(self, output_path, result):
        output_file = open(output_path, 'w+')

        content = str(self.weights.size(0)) + "\t" + str(self.weights.size(1)) + "\n"

        for i, relevance in enumerate(self.relevances.cpu()):
            content += str(i) + "\t" + "\t".join(map(str, relevance.numpy())) + "\n"

        result_text = result.to_string(header=False, index=False)
        result_text = re.sub('\n +', '\n', result_text)
        result_text = re.sub(' +', '\t', result_text)

        content += result_text

        output_file.write(content)

        output_file.close()


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
        calculate_mean = True if batch_size > 1 else False

        if self.weights is None:
            self.initialize_map(x, y, calculate_mean)

        elif batch_size > 1:
            x_sup, y_sup = x[y != self.no_class], y[y != self.no_class]
            if x_sup.size(0) > 0:
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

    def update_map_sup(self, w, y):
        batch_size = w.size(0)

        if batch_size > 1:
            self.batch_update_map_sup(w, y)
        else:
            activations = self.activation(w)[0]
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
                # print("ADD BATCH MAP SUP")
                if indices_geq_at is not None:  # duas amostras de rotulos diferentes ganharam pro mesmo nodo (que pode ter um rotulo, ou nao), o que fazer?
                    new_w, new_y = self.group_data_by_mean(grouped_w[indices_geq_at], grouped_y[indices_geq_at])

                    if new_w.size(0) == 1:
                        self.wins[node] += 1
                        self.classes[node] = new_y
                        self.update_all_connections()
                        self.update_node(new_w, self.e_b, node)
                        self.update_neighbors(new_w, node)
                    # print("ADD NEWSIZE")
                    else:
                        instances = new_w.size(0) - 1
                        if self.weights.size(0) + instances < self.max_node_number:
                            new_ind = self.clone_node(node, instances)

                            self.wins[new_ind] += 1
                            self.classes[new_ind] = new_y
                            self.update_all_connections()
                            self.update_node(new_w, self.e_b, new_ind)

                            self.update_neighbors(new_w, new_ind)
            # print("!!!!!ADD BATCH MAP SUP")
            different_class_nodes = (class_criterion == 0).nonzero()
            if different_class_nodes.size(0) > 0:
                different_class_nodes = different_class_nodes.squeeze(1)

                self.handle_different_class(local_activations[different_class_nodes],
                                            grouped_w[different_class_nodes],
                                            grouped_y[different_class_nodes])

    # print("HANDLE DIFFERENT CLASS")

    def clone_node(self, node, instances):
        # print("CLONE NODE")
        indices = torch.range(0, instances - 1, device=self.device, dtype=torch.long) + self.weights.size(0)

        weights = torch.stack([self.weights[node]] * instances)
        cls = self.classes[node]

        self.add_node(weights, torch.stack([cls] * instances))

        self.moving_avg[indices] = torch.tensor(self.moving_avg[node])
        self.relevances[indices] = torch.tensor(self.relevances[node])
        self.neighbors[indices] = torch.tensor(self.neighbors[node])
        self.wins[indices] = torch.tensor(self.wins[node])

        return torch.cat((indices, node.unsqueeze(0))).sort()[0]

    def handle_different_class(self, activations, w, y):
        new_winners, wrong_winners = self.next_winners(activations, y)
        if wrong_winners is None and new_winners is None:
            new_w, new_y = self.group_data_by_mean(w, y)
            self.add_node(new_w, new_y)
        else:
            no_new_winners = (new_winners == -1).nonzero()
            # print(no_new_winners, no_new_winners.size(0))
            if no_new_winners.size(0) > 0:  # ax1xb = axb
                no_new_winners = no_new_winners.squeeze(1)
                new_w, new_y = self.group_data_by_mean(w[no_new_winners], y[no_new_winners])
                if self.weights.size(0) + new_w.size(0) < self.max_node_number:
                    self.add_node(new_w, new_y)
            valid_new_winners = (new_winners != -1).nonzero()
            if valid_new_winners.size(0) > 0:
                valid_new_winners = valid_new_winners.squeeze(1)

                curr_winners = new_winners[valid_new_winners]
                curr_wrong_winners = wrong_winners[valid_new_winners]

                for new_w, new_y, new_winner, wrong_winner in zip(w[valid_new_winners], y[valid_new_winners],
                                                                  curr_winners, curr_wrong_winners):
                    self.update_node(new_w, -self.e_push, wrong_winner)
                    self.wins[new_winner] += 1
                    self.update_node(new_w, self.e_b, new_winner)
                    self.update_neighbors(new_w, new_winner)

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

    def add_node(self, w, y=None):
        if y is None:
            y = torch.full((w.size(0),), self.no_class, dtype=self.no_class.dtype, device=self.device)

        self.classes = torch.cat((self.classes, y.to(self.device)))

        super(SSSOM, self).add_node(w, y)

    def update_connections(self, node_index):
        dists = self.relevance_distances(self.relevances[node_index], self.relevances)
        dists_connections = (dists < self.minwd)

        stacked_classes = torch.stack([self.classes] * node_index.size(0))
        stacked_classes_transposed = stacked_classes.t()
        classes_connections = (stacked_classes == self.no_class) | (stacked_classes_transposed == self.no_class) | (
                stacked_classes.t() == self.classes[node_index]).t()
        connections = dists_connections & classes_connections

        for node, connection in zip(node_index, connections):
            self.neighbors[node] = connection
            self.neighbors[node][node] = 0

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
        classes_connections = (stacked_classes == self.no_class) | (stacked_classes_transposed == self.no_class) | (
                stacked_classes == stacked_classes_transposed)

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
