import torch
import numpy as np
import scipy.sparse as sp


def prepreocess_adj(adj, is_sparse=False):
    """preprocessing of adjacency matrix for simple GCN model 
    and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return torch.from_numpy(adj_normalized.A).float()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """convert a scipy.sparse matrix to a torch.Tensor format."""
    # TODO
    pass


def normalize_adj(adj):
    """normalize the adjacency matrix symmetrically.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(axis=1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

