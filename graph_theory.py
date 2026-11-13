"""Compute exploratory weighted graph measures from whole-team IBC."""
# Original author: Chen Min

import numpy as np
import pickle
import networkx as nx
from bct import efficiency_wei, clustering_coef_wu, randmio_und


def calculate_global_efficiency(G):
    """Compute weighted global efficiency."""
    adj_matrix = nx.to_numpy_array(G, weight='weight')
    return efficiency_wei(adj_matrix)


def calculate_smallworldness(G, n_rand=20):
    """Compare clustering and inverse efficiency with random rewired graphs."""
    adj = nx.to_numpy_array(G, weight='weight')

    C_real = np.mean(clustering_coef_wu(adj))
    L_real = 1 / efficiency_wei(adj)

    C_rand_list, L_rand_list = [], []
    for _ in range(n_rand):
        adj_rand, _ = randmio_und(adj, itr=10)
        C_rand = np.mean(clustering_coef_wu(adj_rand))
        L_rand = 1 / efficiency_wei(adj_rand)
        C_rand_list.append(C_rand)
        L_rand_list.append(L_rand)

    gamma = C_real / np.mean(C_rand_list)
    lambda_ = L_real / np.mean(L_rand_list)
    sigma = gamma / lambda_
    return sigma


def calculate_leader_centrality(G):
    """Average the weighted degrees of the leader's channel nodes."""
    leader_nodes = [n for n in G.nodes if n.startswith('P0_')]
    leader_degrees = []
    for node in leader_nodes:
        degree = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
        leader_degrees.append(degree)

    return np.mean(leader_degrees)


def calculate_clustering(G):
    """Compute the mean weighted clustering coefficient."""
    adj = nx.to_numpy_array(G, weight='weight')
    cc = clustering_coef_wu(adj)
    return np.nanmean(cc)


def add_edges_from_index(G, C_index_pair, electrodes, thre=95):
    threshold = np.percentile(C_index_pair, thre)

    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for pair_idx, (p1, p2) in enumerate(pairs):

        current_strength = C_index_pair[pair_idx]

        rows, cols = np.where(current_strength >= threshold)
        for i, j in zip(rows, cols):
            weight = current_strength[i][j]
            node1 = f"P{p1}_{electrodes[i]}"
            node2 = f"P{p2}_{electrodes[j]}"
            G.add_edge(node1, node2, weight=weight)

    return G


def bulid_graph(C_index_pair, thre):
    """Build a four-person graph using a percentile threshold from 0 to 100."""

    electrodes = [
        'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5',
        'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6',
        'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2'
    ]

    participants = ['P0', 'P1', 'P2', 'P3']

    node_names = [f"{participant}_{electrode}" for participant in participants for electrode in electrodes]

    G = nx.Graph()
    G.add_nodes_from(node_names)

    G = add_edges_from_index(G, C_index_pair, electrodes, thre)

    return G


def solove_index(G):
    """Return global efficiency, small-worldness, leader centrality and clustering."""
    return [
        calculate_global_efficiency(G),
        calculate_smallworldness(G),
        calculate_leader_centrality(G),
        calculate_clustering(G),
    ]

#%% main

if __name__=='__main__':
    np.random.seed(2)

    with open('./results/C_index.pkl', 'rb') as file_obj:
        C_index = pickle.load(file_obj)

    thre = 95
    index_all = []
    for i in range(C_index.shape[0]):

        index_group = []
        for j in range(C_index.shape[1]):

            index_index = []
            for m in range(C_index.shape[2]):

                index_fre = []
                for n in range(C_index.shape[3]):
                    C_index_pair = C_index[i, j, m, n]
                    G = bulid_graph(C_index_pair, thre)
                    index = solove_index(G)
                    index_fre.append(index)

                index_index.append(index_fre)
            index_group.append(index_index)
        index_all.append(index_group)

    index_all = np.array(index_all)

    with open('./results/index_all_95.pkl', 'wb') as f:
        pickle.dump(index_all, f)
