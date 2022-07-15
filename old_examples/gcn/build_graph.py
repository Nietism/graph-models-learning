import math
import random
import numpy as np

import scipy.sparse as sp
# coo_matrix --> a sparse matrix in COOrdinate format
# csr_matrix --> Compressed Sparse Row matrix
from scipy.sparse import coo_matrix, csr_matrix

import torch
import dgl
from dgl.data import CoraGraphDataset
from utils import prepreocess_adj


class GraphBuild(object):
    """define a class to build 
    """
    def __init__(self):
        self.graph = self.build_graph_test()
        self.adj = self.get_adj(self.graph)
        self.features = self.init_node_features(self.graph)

    def build_graph_test(self):
        """a dummy graph just for testing
        """
        src_nodes = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6])
        dst_nodes = torch.tensor([1, 2, 0, 2, 0, 1, 3, 4, 5, 6, 2, 3, 3, 3])
        graph = dgl.graph((src_nodes, dst_nodes))
        graph.edata["w"] = torch.ones(graph.num_edges())# edges weights if edges has else 1
        return graph

    def convert_symmetric(self, X, sparse=True):
        """add symmetric edges
        """
        if sparse:
            X += X.T - sp.diags(X.diagonal())
        else:
            X += X.T - np.diag(X.diagonal())


    def add_self_loop(self, graph):
        """add self loop for the graph
        """
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        return graph

    def get_adj(self, graph):
        graph = self.add_self_loop(graph)
        graph.edata["w"] = torch.ones(graph.num_edges()) # edges weights if edges has else 1
        adj = coo_matrix(
            (graph.edata["w"], (graph.edges()[0], graph.edges()[1])),
            shape=(graph.num_nodes(), graph.num_nodes())
        )

        # add symmetrix edges
        adj = self.convert_symmetric(adj, sparse=True)
        # add normalize and transform matrix to torch.Tensor shape
        adj = prepreocess_adj(adj, is_sparse=True)

        return adj

if __name__ == "__main__":
    GraphSet = GraphBuild()