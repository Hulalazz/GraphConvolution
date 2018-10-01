"""Data loader and preprocessors for the data publised by Kipf & Welling (2017).

Author: Su Wang.
Modified from: https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""

import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys


def data_mask(index, length):
    """Return a multi-hot vector with the 1 at specified index. Used to select rows from matrix.
    
    Args:
        index: specified index for the 1 cells.
        length: the length of the mask.
    Returns:
        The multi-hot vector with 0->False, 1->True.
    """
    mask = np.zeros(length)
    mask[index] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(file_path):
    """Read the index file to get the indices of the test nodes.
    
    Args:
        file_path: path to a file which is a list of indices.
    Returns:
        The indices as a list.
    """
    indices = []
    for line in open(file_path):
        indices.append(int(line.strip()))
    return indices


def load_data(data_dir, data_name):
    """Loads input data from data directory: citeseer, cora, or pubmed.
    
    Args:
        data_dir: under this folder ...
            ind.data_name.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
            ind.data_name.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
            ind.data_name.allx => the feature vectors of both labeled and unlabeled training instances
                (a superset of ind.data_name.x) as scipy.sparse.csr.csr_matrix object;
            ind.data_name.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
            ind.data_name.ty => the one-hot labels of the test instances as numpy.ndarray object;
            ind.data_name.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
            ind.data_name.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
                object;
            ind.data_name.test.index => the indices of test instances in graph, for the inductive setting as list object.
            All objects above must be saved using python pickle module.
        data_name: "citeseer", "cora", or "pubmed".
    Returns: 
        adj_matrix: a <number_data, number_data> sparse matrix (scipy.sparse.csr.csr_matrix).
        features: a <number_data, number_features> sparse matrix (scipy.sparse.lil.lil_matrix).
        y_train: a <number_data, number_classes> numpy ndarray, one-hot rows for classes.
            the rows not corresponding to training data are all-0. 
        y_valid: same as y_train, with rows not corresponding to validation data all-0.
        y_test: same as y_train, with rows not corresponding to test data all-0.
        train_mask: <number_data, 1> numpy ndarray, rows are boolean, selecting train rows.
        valid_mask: same as train_mask, rows selecting validation rows.
        test_mask: same as train_mask, rows selecting test_rows.
    """  
    assert data_name in ["citeseer", "cora", "pubmed"]
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for name in names:
        with open(data_dir + "ind." + data_name + "." + name, "rb") as f:
            objects.append(pkl.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_index_reorder = parse_index_file(data_dir + "ind." + data_name + ".test.index") # unordered indices.    
    test_index_range = np.sort(test_index_reorder) # ordered indices.
    # Fix citeseer data by assigning 0 vectors to isolated notes.
    if data_name == "citeseer":
        test_index_range_full = range(min(test_index_reorder), max(test_index_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_index_range_full), x.shape[1])) # linked list based matrix.
        tx_extended[test_index_range - min(test_index_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_index_range_full), y.shape[1]))
        ty_extended[test_index_range - min(test_index_range), :] = ty
        ty = ty_extended
    # Load features
    features = sp.vstack((allx, tx)).tolil() # to linked list based matrix.
    features[test_index_reorder, :] = features[test_index_range, :] # sort the test nodes related rows.
    # Load adjacent matrix
    adj_matrix = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) # number_nodes x number_nodes shape.
    # Load labels
    labels = np.vstack((ally, ty)) # number_nodes x number_classes.
    labels[test_index_reorder, :] = labels[test_index_range, :] # sort the test nodes related rows.
    # Make mask for row selection
    indices_test = test_index_range.tolist()
    indices_train = range(len(y))
    indices_valid = range(len(y), len(y)+500) # select the first 500 nodes after the labeled nodes as validation.
    train_mask = data_mask(indices_train, labels.shape[0])
    valid_mask = data_mask(indices_valid, labels.shape[0])
    test_mask = data_mask(indices_test, labels.shape[0])
    # Make containers for predictions
    y_train = np.zeros(labels.shape)
    y_valid = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_valid[valid_mask, :] = labels[valid_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    # <mask>[:, newaxis]: make shape [number_data, 1] for broadcase multiplication with
    #   the data, which has the shape [number_data, number_classes].
    return adj_matrix, features, y_train, y_valid, y_test, \
           train_mask[:, np.newaxis], valid_mask[:, np.newaxis], test_mask[:, np.newaxis] 


def get_selected_indices(mask):
    """Return the indices of the data points selected for slicing train/validation/test.
    
    Args:
        mask: a <number_data, 1> matrix. Each cell is [True]/[False], indicating
            whether the correponding data point is selected.
    Returns:
        The indices of the selected data points as a list.
    """
    return [index for index, entry in enumerate(mask) if entry[0]]


def to_categorical(one_hot_matrix):
    """Convert the one-hot-rowed matrix into categorical-rowed matrix.
    
    Args:
        one_hot_matrix: of shape <number_data, number_classes> with one-hot rows.
    Returns:
        A <number_data, 1> 
    """
    number_classes = one_hot_matrix.shape[1]
    categorical = []
    for row in one_hot_matrix:
        index = np.where(row==1)[0]
        # Handles unlabeled data points, which correponds to all-0 row.
        if row.sum() == 0:  
            categorical.append(number_classes)
        else:
            categorical.append(index[0])
    return np.array(categorical)    


def to_A_tilde(A):
    """Add self-connection (Kipf & Welling, 2017, section 2) to adjacency matrix."""
    return A + sp.eye(A.shape[0])


def to_A_hat(A_tilde):
    """Normalize adjacency matrix (with self-connections), (Kipf & Welling, 2017, section 3.1).
    
    NB: the matrix inversion here is on diagonal matrix. The inverse of a diagonal matrix
        has its diagonal cells as the reciprocal of that of the original diagonal matrix.
    """
    A_tilde = sp.coo_matrix(A_tilde)
    rowsum = np.array(A_tilde.sum(axis=1)) # compute diagonal values for D.
    D_inv_sqrt = np.power(rowsum, -0.5).flatten() # compute inverse square root of D.
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0. # convert inf values to 0.
    D_inv_sqrt = sp.diags(D_inv_sqrt) # convert vectorized diagonal values to a diagonal matrix.
    return A_tilde.dot(D_inv_sqrt).transpose().dot(D_inv_sqrt).tocoo() # formula for A_hat.

