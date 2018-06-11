# Author: Pedro Braga <phmb4@cin.ufpe.br>.

import torch as torch
import torch.nn as nn
import pandas as pd
import itertools


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
        # sssom = torch.nn.DataParallel(sssom)
        x = x.to(self.device)
        y = y.to(self.device)

        if self.weights is None:
            self.initialize_map(x, y)
        else:
            self.update_map(x)

        if self.step >= self.age_wins:
            self.remove_loosers()
            self.wins = torch.zeros(self.wins.size(0), device=self.device)
            self.update_all_connections()

            self.step = 0

        self.step += 1

    def update_map(self, w):
        activations = self.activation(w)
        ind_max = torch.argmax(activations).item()

        if activations[ind_max].item() < self.a_t.item() and self.weights.size(0) < self.max_node_number:
            self.add_node(w)
        elif activations[ind_max].item() >= self.a_t.item():
            self.wins[ind_max] += 1
            self.update_node(w, self.e_b, ind_max)

            neighbors = self.neighbors[ind_max].nonzero()
            if neighbors.size(0) > 0:
                neighbors_ind = neighbors.squeeze(1)
                self.update_node(w, self.e_n, neighbors_ind)

    def activation(self, w):
        dists = self.weighted_distance(w)
        relevances_sum = torch.sum(self.relevances, 1)

        return torch.div(relevances_sum, torch.add(torch.add(relevances_sum, dists), 1e-7))

    def initialize_map(self, w, y=None):
        self.weights = torch.tensor(w)
        self.moving_avg = torch.zeros(1, self.dim, device=self.device)
        self.relevances = torch.ones(1, self.dim, device=self.device)

        self.neighbors = torch.zeros(1, 1, dtype=torch.uint8, device=self.device)
        self.wins = torch.zeros(1, device=self.device)

    def add_node(self, w, y=None):
        self.weights = torch.cat((self.weights, w), 0)

        zeros = torch.zeros(1, self.dim, device=self.device)
        ones = torch.ones(1, self.dim, device=self.device)

        self.moving_avg = torch.cat((self.moving_avg, zeros), 0)
        self.relevances = torch.cat((self.relevances, ones), 0)

        self.neighbors = torch.cat((self.neighbors,
                                    torch.zeros(self.neighbors.size(0), 1, dtype=torch.uint8, device=self.device)), 1)
        self.neighbors = torch.cat((self.neighbors,
                                    torch.zeros(1, self.neighbors.size(1), dtype=torch.uint8, device=self.device)), 0)
        self.update_connections(self.weights.size(0) - 1)

        self.wins = torch.cat((self.wins, torch.zeros(1, device=self.device)))

    def update_connections(self, node_index):
        dists = self.relevance_distances(self.relevances[node_index], self.relevances)
        dists_connections = (dists < self.minwd)
        dists_connections[node_index] = 0

        self.neighbors[node_index] = dists_connections

    def update_node(self, w, lr, index):
        distance = torch.abs(torch.sub(w, self.weights[index]))
        self.moving_avg[index] = torch.mul(lr * self.dsbeta, distance) + torch.mul(1 - lr * self.dsbeta, self.moving_avg[index])

        if torch.is_tensor(index):
            maximum = torch.max(self.moving_avg[index], 1)[0].unsqueeze(1)
            minimum = torch.min(self.moving_avg[index], 1)[0].unsqueeze(1)
            avg = torch.mean(self.moving_avg[index], 1).unsqueeze(1)
        else:
            maximum = torch.max(self.moving_avg[index])
            minimum = torch.min(self.moving_avg[index])
            avg = torch.mean(self.moving_avg[index])

        one = torch.tensor(1, dtype=torch.float, device=self.device)

        self.relevances[index] = torch.div(one, one + torch.exp(torch.div(torch.sub(self.moving_avg[index], avg), torch.mul(self.eps_ds, torch.sub(maximum, minimum)))))
        self.relevances[self.relevances != self.relevances] = 1.  # if (max - min) == 0 then set to 1

        self.weights[index] = torch.add(self.weights[index], torch.mul(lr, torch.sub(w, self.weights[index])))

    def remove_loosers(self):
        remaining_indexes = (self.wins >= self.step * self.lp).nonzero().squeeze(1)

        self.weights = self.weights[remaining_indexes]
        self.moving_avg = self.moving_avg[remaining_indexes]
        self.relevances = self.relevances[remaining_indexes]

        self.neighbors = []
        self.wins = self.wins[remaining_indexes]

        return remaining_indexes

    def update_all_connections(self):
        dists_stacked = torch.stack([self.relevances] * self.relevances.size(0))
        dists_stacked_transposed = dists_stacked.transpose(0, 1)
        dists = self.relevance_distances(dists_stacked, dists_stacked_transposed)
        dists_connections = dists < self.minwd

        self.neighbors = (1 - torch.eye(self.weights.size(0), dtype=torch.uint8, device=self.device)) * dists_connections

    def weighted_distance(self, w):
        sub = torch.sub(w, self.weights)
        qrt = torch.pow(sub, 2)
        weighting = torch.mul(self.relevances, qrt)
        summation = torch.sum(weighting, 1)

        return summation
        # return torch.sqrt(summation)

    def relevance_distances(self, x1, x2):
        fabs = torch.abs(torch.sub(x1, x2))
        return torch.sum(fabs, fabs.dim() - 1)

    def finish_map(self, dataloader):
        dataiter = itertools.cycle(dataloader)

        self.age_wins_cycle(dataiter)

        self.max_node_number = self.weights.size(0)

        data = next(dataiter)
        self.forward(data[0], data[1])

        self.age_wins_cycle(dataiter)

    def age_wins_cycle(self, dataiter):
        while self.step != 1:
            data = next(dataiter)
            self.forward(data[0], data[1])

    def fit(self, dataloader):
        for epoch in range(self.epochs):
            for i, data in enumerate(dataloader):
                self.forward(data[0], data[1])

        self.finish_map(dataloader)

    def cluster(self, dataloader, is_subspace, filter_noise):
        clustering = pd.DataFrame(columns=['sample_ind', 'cluster'])
        for i, data in enumerate(dataloader, 0):
            activations = self.activation(data[0].to(self.device))
            ind_max = torch.argmax(activations).item()

            if filter_noise and activations[ind_max] < self.a_t:
                continue

            if not is_subspace:
                clustering = clustering.append({'sample_ind': i, 'cluster': ind_max}, ignore_index=True)
            else:
                x = 1

        return clustering

    def write_output(self, output_path, result):
        output_file = open(output_path, 'w+')

        content = str(self.weights.size(0)) + "\t" + str(self.weights.size(1)) + "\n"

        for i, relevance in enumerate(self.relevances.cpu()):
            content += str(i) + "\t" + "\t".join(map(str, relevance.numpy())) + "\n"

        result_text = result.to_string(header=False, index=False)
        result_text = result_text.replace("\n ", "\n")
        result_text = result_text.replace("  ", "\t")

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

        if self.weights is None:
            self.initialize_map(x, y)
        elif torch.eq(y, self.no_class):
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
        activations = self.activation(w)
        ind_max = torch.argmax(activations).item()

        if self.classes[ind_max] == y or self.classes[ind_max] == self.no_class:
            if activations[ind_max] < self.a_t and self.weights.size(0) < self.max_node_number:
                self.add_node(w, y)

            elif activations[ind_max] >= self.a_t:
                self.wins[ind_max] += 1
                self.classes[ind_max] = y
                self.update_connections(ind_max)
                self.update_node(w, self.e_b, ind_max)

                neighbors = self.neighbors[ind_max].nonzero()
                if neighbors.size(0) > 0:
                    neighbors_ind = neighbors.squeeze(1)
                    self.update_node(w, self.e_n, neighbors_ind)
        else:
            self.handle_different_class(activations, w, y)

    def handle_different_class(self, activations, w, y):
        wrong_winner, new_winners = self.next_winners(activations, y)

        if wrong_winner is not None and new_winners is not None:
            winner = new_winners[0]
            self.update_node(w, -self.e_push, wrong_winner.item())
            self.wins[winner] += 1

            self.update_node(w, self.e_b, winner.item())

            neighbors = self.neighbors[winner].nonzero()
            if neighbors.size(0) > 0:
                neighbors_ind = neighbors.squeeze(1)
                self.update_node(w, self.e_n, neighbors_ind)

        elif self.weights.size(0) < self.max_node_number:
            self.add_node(w, y)

    def next_winners(self, activations, y):
        wrong_max = None
        winners = None

        sorted_activations, indices = torch.sort(activations, descending=True)

        indices = indices[sorted_activations >= self.a_t]

        if indices.size(0) > 0:
            wrong_max = indices[0]

            no_classes = self.classes[indices] == self.no_class
            same_classes = self.classes[indices] == y

            winners = indices[no_classes | same_classes]

            if winners.size(0) == 0:
                winners = None

        return wrong_max, winners

    def initialize_map(self, w, y=None):
        super(SSSOM, self).initialize_map(w, y)

        if y is None:
            y = self.no_class

        self.classes = y.to(self.device)

    def add_node(self, w, y=None):
        if y is None:
            y = self.no_class

        self.classes = torch.cat((self.classes, y.to(self.device)))

        super(SSSOM, self).add_node(w, y)

    def update_connections(self, node_index):
        dists = self.relevance_distances(self.relevances[node_index], self.relevances)
        dists_connections = (dists < self.minwd)
        dists_connections[node_index] = 0

        if not torch.eq(self.classes[node_index], self.no_class):
            classes_connections = (self.classes == self.no_class) | (self.classes == self.classes[node_index])
            self.neighbors[node_index] = dists_connections & classes_connections
        else:
            self.neighbors[node_index] = dists_connections

    def remove_loosers(self):
        remaining_indexes = super(SSSOM, self).remove_loosers()
        self.classes = self.classes[remaining_indexes]

        return remaining_indexes

    def update_all_connections(self):
        dists_stacked = torch.stack([self.relevances] * self.relevances.size(0))
        dists_stacked_transposed = dists_stacked.transpose(0, 1)
        dists = self.relevance_distances(dists_stacked, dists_stacked_transposed)
        dists_connections = dists < self.minwd

        classes_stacked = torch.stack([self.classes.unsqueeze(0)] * self.classes.unsqueeze(0).size(0))
        classes_stacked_transposed = classes_stacked.transpose(0, 1)
        classes_connections = (classes_stacked == self.no_class) | (classes_stacked == classes_stacked_transposed)

        connections = dists_connections & classes_connections

        self.neighbors = (1 - torch.eye(self.weights.size(0), dtype=torch.uint8, device=self.device)) * connections.squeeze(0)

    def cluster_classify(self, dataloader, is_subspace, filter_noise):
        clustering_classify = pd.DataFrame(columns=['sample_ind', 'cluster', 'class'])
        for i, data in enumerate(dataloader, 0):
            activations = self.activation(data[0].to(self.device))
            ind_max = torch.argmax(activations).item()

            if filter_noise and activations[ind_max] < self.a_t:
                continue

            if not is_subspace:

                if self.classes[ind_max] == self.no_class:
                    _, indices = torch.sort(activations, descending=True)

                    classes = self.classes[indices] != self.no_class

                    winners = indices[classes]

                    if winners.size(0) > 0:
                        ind_max = winners[0]

                clustering_classify = clustering_classify.append({'sample_ind': i,
                                                                  'cluster': ind_max,
                                                                  'class': self.classes[ind_max].item()},
                                                                 ignore_index=True)
            else:
                x = 1

        return clustering_classify
