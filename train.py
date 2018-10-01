"""Train the Graph Convolutional Networks at described in Kipf & Welling (2017).

Author: Su Wang.
"""


import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn

from models import *
from utils import *

CUDA = torch.cuda.is_available()


def to_tensor(raw_inputs, tensor_type):
    """numpy ndarray or sparse matrix to torch tensor.
    
    Args: 
        raw_inputs: <number_data, number_features> shaped sparse matrix or numpy ndarray.
        tensor_type: torch tensor type.
    Returns:
        tensor: a torch tensor of the same shape as raw_input with type tensor_type.
    """
    if type(raw_inputs) != np.ndarray:
        raw_inputs = raw_inputs.toarray()
    tensor = Variable(torch.Tensor(raw_inputs).type(tensor_type))
    if CUDA:
        return tensor.cuda()
    return tensor


def compute_loss(model, criterion, l2_coefficient, out, gold):
    """Compute loss.
    
    Args:
        model: a GCN object.
        criterion: torch.nn.CrossEntropyLoss.
        l2_coefficient: L2 regularization coefficient.
        out: model prediction (unnormalized), shape = <number_data, number_classes>.
        gold: true labels, shape = <number_data, >.
    Returns:
        Loss.
    """
    loss = criterion(out, gold)
    layer_1_parameters = torch.cat([param.view(-1) for param in model.layer_1.parameters()])
    layer_1_l2_loss = l2_coefficient * torch.norm(layer_1_parameters, 2)
    return loss + layer_1_l2_loss


def train_gcn(adj_matrix, features, 
              y_train, y_valid, y_test,
              train_mask, valid_mask, test_mask,
              hidden_size_1=16, learning_rate=1e-2, 
              drop_prob=0.5, l2_coefficient=5e-4,
              early_stopping_threshold=0.001,
              number_iterations=100, print_every=5):
    """Training a Graph Convolutional Network (Kipf & Welling, 2017).
    
    Args:
        adj_matrix: a <number_data, number_data> sparse matrix (scipy.sparse.csr.csr_matrix).
        features: a <number_data, number_features> sparse matrix (scipy.sparse.lil.lil_matrix).
        y_train: a <number_data, number_classes> numpy ndarray, one-hot rows for classes.
            the rows not corresponding to training data are all-0. 
        y_valid: same as y_train, with rows not corresponding to validation data all-0.
        y_test: same as y_train, with rows not corresponding to test data all-0.
        train_mask: <number_data, 1> numpy ndarray, rows are boolean, selecting train rows.
        valid_mask: same as train_mask, rows selecting validation rows.
        test_mask: same as train_mask, rows selecting test_rows.
        hidden_size_1: the embedding size of the first layer.
        learning_rate: float learning rate for the Adam optimizer.
        drop_prob: dropout rate.
        l2_coefficient: L2 regularization coefficient.
        early_stopping_threshold: stopping threshold on training accuracy.
        number_iterations: number of iterations.
        print_every: validation frequency.
    """
    
    A_tilde = to_A_tilde(adj_matrix)
    A_hat = to_tensor(to_A_hat(A_tilde), torch.FloatTensor)
    X = to_tensor(features, torch.FloatTensor)
    
    in_features = X.shape[1]
    hidden_size_2 = y_train.shape[1] + 1 # one more for all-0 rows.    
    
    y_train = to_tensor(to_categorical(y_train * train_mask), torch.LongTensor)
    y_valid = to_tensor(to_categorical(y_valid * valid_mask), torch.LongTensor)
    y_test = to_tensor(to_categorical(y_test * test_mask), torch.LongTensor)
    number_train = train_mask.sum()
    number_valid = valid_mask.sum()
    number_test = test_mask.sum() 
    train_indices = get_selected_indices(train_mask)
    valid_indices = get_selected_indices(valid_mask)
    test_indices = get_selected_indices(test_mask)
    
    gcn = GCN(in_features, hidden_size_1, hidden_size_2, drop_prob)
    if CUDA:
        gcn = gcn.cuda()
        A_hat, X = A_hat.cuda(), X.cuda()
        y_train, y_valid, y_test = y_train.cuda(), y_valid.cuda(), y_test.cuda()
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=learning_rate)
    softmax = nn.Softmax(dim=-1)
    
    for i in range(number_iterations):
        if i != 0 and i % print_every == 0:
            mode = "eval"
        else:
            mode = "train"
        if mode == "train":
            gcn.train()
            out = gcn(A_hat, X)
            # NB: arg1 has type .FloatTensor, arg2 has type .LongTensor.
            train_loss = compute_loss(gcn, criterion, l2_coefficient,
                                      out[train_indices], y_train[train_indices])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        else:
            gcn.eval()
            out = gcn(A_hat, X)
            # NB: arg1 has type .FloatTensor, arg2 has type .LongTensor.
            train_loss = compute_loss(gcn, criterion, l2_coefficient,
                                      out[train_indices], y_train[train_indices])
            print("Iteration " + str(i) + ":\n")
            if CUDA:
                train_loss = train_loss.cpu()
            print("Train loss =", train_loss.data.numpy()[0], "(at step "+str(i)+")")
            result = softmax(out)
            _, predictions = torch.max(result.data, 1)
            number_correct_train = predictions[train_indices] \
                .eq(y_train[train_indices].data).sum()
            number_correct_valid = predictions[valid_indices] \
                .eq(y_valid[valid_indices].data).sum()
            number_correct_test = predictions[test_indices] \
                .eq(y_test[test_indices].data).sum()      
            accuracy_train = number_correct_train / number_train
            accuracy_valid = number_correct_valid / number_valid
            accuracy_test = number_correct_test / number_test
            print("Train/Valid/Test accuracy: %.4f | %.4f | %.4f\n" % (accuracy_train,
                                                                       accuracy_valid,
                                                                       accuracy_test))
            if accuracy_train + early_stopping_threshold >= 1.0:
                print("Training accuracy reaches 1.0, training done!")
                break
    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--hidden_size_1", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--drop_prob", type=float)
    parser.add_argument("--l2_coefficient", type=float)
    parser.add_argument("--number_iterations", type=int)
    parser.add_argument("--print_every", type=int)
    args = parser.parse_args()
    
    adj_matrix, features, y_train, y_valid, y_test, \
    train_mask, valid_mask, test_mask = load_data(args.data_dir, args.data_name)
    
    train_gcn(adj_matrix, features, 
              y_train, y_valid, y_test,
              train_mask, valid_mask, test_mask,
              args.hidden_size_1, args.learning_rate, 
              args.drop_prob, args.l2_coefficient,
              args.early_stopping_threshold,
              args.number_iterations, args.print_every)    