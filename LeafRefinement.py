from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree
import copy
from tqdm import tqdm

import torch
import torch.nn as nn

def get_subtree(sk_model, max_depth):
    copyed_model = copy.deepcopy(sk_model)
    sk_tree = copyed_model.tree_
    left = sk_tree.children_left
    right = sk_tree.children_right
    nodes = [(0, 0)]
    n_nodes = 1
    while len(nodes) > 0:
        n, depth = nodes.pop(0)

        if depth >= max_depth or (left[n] == _tree.TREE_LEAF and right[n] == _tree.TREE_LEAF):
            left[n] = _tree.TREE_LEAF
            right[n] = _tree.TREE_LEAF
        else:
            if left[n] != _tree.TREE_LEAF:
                nodes.append( (left[n], depth + 1) )
                n_nodes += 1 # Inner node 

            if right[n] != _tree.TREE_LEAF:
                nodes.append( (right[n], depth + 1) ) 
                n_nodes += 1 # Inner node 

    copyed_model.tree_.children_left[:] = left[:]
    copyed_model.tree_.children_right[:] = right[:]
    return copyed_model, n_nodes

class Tree():
    def __init__(self, sk_model, max_depth):
        self.model = copy.deepcopy(sk_model)
        sk_tree = self.model.tree_

        if max_depth is not None:
            left = sk_tree.children_left
            right = sk_tree.children_right
            nodes = [(0, 0)]
            n_nodes = 1
            while len(nodes) > 0:
                n, depth = nodes.pop(0)

                if depth >= max_depth or (left[n] == _tree.TREE_LEAF and right[n] == _tree.TREE_LEAF):
                    left[n] = _tree.TREE_LEAF
                    right[n] = _tree.TREE_LEAF
                else:
                    if left[n] != _tree.TREE_LEAF:
                        nodes.append( (left[n], depth + 1) )
                        n_nodes += 1 # Inner node 

                    if right[n] != _tree.TREE_LEAF:
                        nodes.append( (right[n], depth + 1) ) 
                        n_nodes += 1 # Inner node 

            self.model.tree_.children_left[:] = left[:]
            self.model.tree_.children_right[:] = right[:]
            self.n_nodes = n_nodes
        else:
            self.n_nodes = self.model.tree_.node_count

        # sk_tree = sk_model.tree_
        # n_classes = sk_model.n_classes_
        # self.leafs = np.array([p / sum(p) for p in sk_tree.value.reshape((-1,n_classes))])
        # self.features = sk_tree.feature
        # self.splits = sk_tree.threshold
        # self.left = [i if i != _tree.TREE_LEAF else None for i in sk_tree.children_left]
        # self.right = [i if i != _tree.TREE_LEAF else None for i in sk_tree.children_right]

    def apply(self,X):
        return self.model.apply(X)

    def predict_proba(self,X):
        # Need to do this because sk learns self.model.predict_proba normalizes the leafs before returning
        idx = self.model.apply(X)
        return self.model.tree_.value[idx,:].squeeze(1)

    def predict(self,X):
        return self.predict_proba(X).argmax(axis=1)

def create_mini_batches(inputs, targets, batch_size, shuffle=False):
    """ Create an mini-batch like iterator for the given inputs / target / data. Shamelessly copied from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    
    Parameters
    ----------
    inputs : array-like vector or matrix 
        The inputs to be iterated in mini batches
    targets : array-like vector or matrix 
        The targets to be iterated in mini batches
    batch_size : int
        The mini batch size
    shuffle : bool, default False
        If True shuffle the batches 
    """
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        
        start_idx += batch_size

        yield inputs[excerpt], targets[excerpt]

# def to_prob_simplex(x):
#     """ Projects the given vector to the probability simplex so that :math:`\\sum_{i=1}^k x_i = 1, x_i \\in [0,1]`. 

#     Reference
#         Weiran Wang and Miguel A. Carreira-Perpinan (2013) Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application. https://arxiv.org/pdf/1309.1541.pdf

#     Parameters
#     ----------
#     x : array-like vector with k entries
#         The vector to be projected.

#     Returns
#     -------
#     u : array-like vector with k entries
#         The projected vector.

#     """
#     if x is None or len(x) == 0:
#         return x
#     u = np.sort(x)[::-1]
    
#     l = None
#     u_sum = 0
#     for i in range(0,len(u)):
#         u_sum += u[i]
#         tmp = 1.0 / (i + 1.0) * (1.0 - u_sum)
#         if u[i] + tmp > 0:
#             l = tmp
    
#     projected_x = [max(xi + l, 0.0) for xi in x]
#     return projected_x

def to_prob_simplex(x, lam = 1.0):
    if x is None or len(x) == 0:
        return x
    u = np.sort(x)[::-1]
    
    l = None
    u_sum = 0
    for i in range(0,len(u)):
        u_sum += u[i]
        tmp = 1.0 / (i + 1.0) * (u_sum - lam)
        if u[i] > tmp:
            l = tmp
    
    #projected_x = [max(xi + l, 0.0) for xi in x]
    projected_x = [max(xi - l, 0.0) for xi in x]
    return projected_x

def prox(w, prox_type, normalize, l_reg, step_size, lam):
    # print("")
    # print("BEFORE: ", w)

    if prox_type == "L0":
        tmp = np.sqrt(2 * l_reg * step_size)
        tmp_w = np.array([0 if abs(wi) < tmp else wi for wi in w])
    elif prox_type == "L1":
        sign = np.sign(w)
        tmp_w = np.abs(w) - l_reg*step_size
        tmp_w = sign*np.maximum(tmp_w,0)
    elif prox_type == "hard-L0":
        top_K = np.argsort(w)[-l_reg:]
        tmp_w = np.array([wi if i in top_K else 0 for i,wi in enumerate(w)])
    else:
        tmp_w = w

    # If set, normalize the weights. Note that we use the support of tmp_w for the projection onto the probability simplex
    # as described in http://proceedings.mlr.press/v28/kyrillidis13.pdf
    # Thus, we first need to extract the nonzero weights, project these and then copy them back into corresponding array
    if normalize and len(tmp_w) > 0:
        nonzero_idx = np.nonzero(tmp_w)[0]
        nonzero_w = tmp_w[nonzero_idx]
        nonzero_w = to_prob_simplex(nonzero_w, lam)
        new_w = np.zeros((len(tmp_w)))
        for i,wi in zip(nonzero_idx, nonzero_w):
            new_w[i] = wi
        # print("AFTER: ", new_w)
        # print("")

        return new_w
    else:
        return tmp_w

class LeafRefinery(nn.Module):

    def __init__(self, epochs, lr, batch_size, optimizer, verbose,loss_function = "mse", loss_type = "upper", l_reg = 1.0, base_forest = {}, l1_strength = 0, pruner = "L1", n_jobs = 1, leaf_refinement=True):#, ensemble_regularizer = "none"):
        super().__init__()

        if n_jobs is not None:
            torch.set_num_threads(n_jobs)

        assert loss_function in ["mse", "nll", "cross-entropy"], "LeafRefinery only supports the {{mse, nll, cross-entropy}} loss but you gave {}".format(loss_function)
        assert lr >= 0, "Learning rate must be positive, but you gave {}".format(lr)
        assert epochs >= 0, "Number of epochs must be positive, but you gave {}".format(epochs)
        assert optimizer in ["sgd", "adam"], "The optimizer must be from {{adam, sgd}}, but you gave {}".format(optimizer)
        # assert ensemble_regularizer is None or ensemble_regularizer in ["none","L0", "L1", "hard-L0", "random"], "Currently only {{none,L0, L1, hard-L0, random}} the ensemble regularizer is supported, but you gave {}".format(ensemble_regularizer)
        
        # if ensemble_regularizer in ["hard-L0", "random"]:
        #     assert l_ensemble_reg >= 1 or l_ensemble_reg == 0, "You chose ensemble_regularizer do be one of {{hard-L0, random}}, but set 0 < l_ensemble_reg < 1 which does not really makes sense. If {{hard-L0, random}} is set, then l_ensemble_reg is the maximum number of estimators in the pruned ensemble, thus likely an integer value >= 1."

        if loss_type == "exact":
            assert 0 <= l_reg <= 1, "You set loss_type to exact. In this case l_reg should be from [0,1], but you supplied l_reg = {}".format(l_reg)

        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.verbose = verbose
        self.loss_function = loss_function
        self.loss_type = loss_type
        self.l_reg = l_reg
        self.base_forest = base_forest
        # self.normalize_weights = normalize_weights
        #self.ensemble_regularizer = ensemble_regularizer
        self.normalize_weights = False
        # self.ensemble_regularizer = "L1"
        self.l_ensemble_reg = l1_strength
        self.pruner = pruner
        self.leaf_refinement = leaf_refinement
        if pruner == "L1":
            self.ensemble_regularizer = "L1"
        elif pruner == "hard-L0":
            self.ensemble_regularizer = "hard-L0"
            self.normalize_weights = True
        else:
            self.ensemble_regularizer = "none"

    def predict_proba(self, X):
        if len(self.trees) == 0:
            y_default = np.array([1.0 / self.n_classes_ for _ in range(self.n_classes_)])
            for _ in X:
                proba.append(y_default)
            return np.array(proba)
        else:
            proba = []
            for h, w in zip(self.trees, self.weights):
                proba.append(w * h.predict_proba(X))
            return np.stack(proba).mean(axis=0)
        
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def _loss(self, pred, target):
        if self.loss_function == "mse":
            target_one_hot = torch.nn.functional.one_hot(target, num_classes = pred.shape[1]).double()
            return torch.nn.functional.mse_loss(pred, target_one_hot)
        elif self.loss_function == "nll":
            return torch.nn.functional.nll_loss(pred, target)
        elif self.loss_function == "cross-entropy":
            return torch.nn.functional.cross_entropy(pred, target)
        else:
            raise ValueError("Unknown loss function set in LeafRefinery")

    def compute_loss(self, fbar, base_preds, target):
        if self.loss_type == "upper":
            n_classes = fbar.shape[1]
            n_preds = fbar.shape[0]
            D = torch.eye(n_classes).repeat(n_preds, 1, 1).double()
        else:
            if self.loss_function == "mse":
                n_classes = fbar.shape[1]
                n_preds = fbar.shape[0]

                eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).double()
                D = 2.0*eye_matrix
            elif self.loss_function == "nll":
                n_classes = fbar.shape[1]
                n_preds = fbar.shape[0]
                D = torch.eye(n_classes).repeat(n_preds, 1, 1).double()
                target_one_hot = torch.nn.functional.one_hot(target, num_classes = n_classes)

                eps = 1e-7
                diag_vector = target_one_hot*(1.0/(fbar**2+eps))
                D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
            elif self.loss_function == "cross-entropy":
                n_preds = fbar.shape[0]
                n_classes = fbar.shape[1]
                f_bar_softmax = nn.functional.softmax(fbar,dim=1)

                D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1)).double()
                diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
                D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
            else:
                # NOTE: We should never reach this code path
                raise ValueError("Invalid combination of mode and loss function in Leaf-refinement.")

        f_loss = self._loss(fbar, target)
        losses = []
        n_estimators = len(base_preds)
        for pred in base_preds:
            diff = pred - fbar
            covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
            div = 1.0/n_estimators * 1.0/2.0 * covar

            i_loss = self._loss(pred, target)

            if self.loss_type == "exact":
                # Eq. (4)
                reg_loss = 1.0/n_estimators * i_loss - self.l_reg * div
            else:
                # Eq. (5) where we scale the ensemble loss with 1.0/self.n_estimators due to the summation later
                reg_loss = 1.0/n_estimators * self.l_reg * f_loss + (1.0 - self.l_reg)/n_estimators * i_loss
            
            losses.append(reg_loss)
        return torch.stack(losses).sum()

    def fit(self, X, Y):
        """Performs SGD using the MSE loss over the leaf nodes of the given trees on the given data. The weights of each tree are respected during optimization but not optimized. 

        Args:
            weights (np.array): The weights of the trees.
            trees (list of Tree): The trees.
            X (2d np.array): The data.
            Y (np.array): The targe.
            epochs (int): The number of epochs SGD is performed.
            lr (float): The learning rate of SGD.
            batch_size (int): The batch size of SGD
            optimizer (str): The optimizer used for optimization. Can be {{"sgd", "adam"}}.
            verbose (bool): If True outputs the loss during optimization.

        Returns:
            list of trees: The refined trees.
        """

        self.forest = RandomForestClassifier(**self.base_forest)
        self.forest.fit(X,Y)
        self.prune(X,Y,self.forest.estimators_, self.forest.classes_, self.forest.n_classes_)

    def prune(self, X, Y, estimators, classes, n_classes):
        self.n_classes_ = n_classes
        self.classes_ = classes

        if self.batch_size > X.shape[0]:
            if self.verbose:
                print("WARNING: The batch size for SGD is larger than the dataset supplied: batch_size = {} > X.shape[0] = {}. Using batch_size = X.shape[0]".format(self.batch_size, X.shape[0]))
            self.batch_size = X.shape[0]

        # To make the following SGD somewhat efficient this code extracts all the leaf nodes and gathers them in an array. To do so it iterates over all trees and all nodes in the trees. Each leaf node is added to the leafs array and the corresponding node.id is stored in mappings. For scikit-learn trees this would be much simpler as they already offer a dedicated leaf field:
        
        # TODO THIS SHOULD BE PERFORMED BY A PRUNER WHICH IS GIVEN BEFOREHAND AND ONE OF THEM IS L1 WHICH IS AS BELOW
        # Experiments:
        # only pruning, no refinement => epochs = 0
        # no pruning, only refinement => random pruning
        # pruning => refinement => 
        # joint via L1
        self.trees = [Tree(e, None) for e in estimators] 
        if self.pruner != "L1" and self.pruner != "hard-L0":
            self.pruner.prune(X,Y,self.trees, classes, n_classes)
            self.trees, weights = self.pruner.estimators_, np.array(self.pruner.weights_)
        else:
            weights = np.repeat(1.0/len(self.trees), len(self.trees))
        n_trees = len(self.trees)
        # if self.ensemble_regularizer == "random":
        #     self.trees = self.trees[:self.l_ensemble_reg]

        # self.trees = []
        # TODO Regularization for DTs just does not work. We either always use all trees or no trees :/
        # node_counts = []
        # for e in self.forest.estimators_:
        #     max_depth = int(np.log2(e.tree_.node_count)) + 1
        #     for d in range(1, max_depth):
        #         self.trees.append(Tree(e, d))
        #         node_counts.append(d)
        # node_counts = torch.as_tensor(node_counts)
        
        # print("STARTING WITH {} TREES".format(n_trees))

        torch_leafs = []
        for h in self.trees:
            tmp = h.model.tree_.value / h.model.tree_.value.sum(axis=(1,2))[:,np.newaxis,np.newaxis]
            torch_leafs.append(nn.Parameter(torch.from_numpy(tmp.squeeze(1))))

        self.torch_leafs = nn.ParameterList(torch_leafs)
        if not self.leaf_refinement:
            for i in range(len(self.torch_leafs)):
                self.torch_leafs[i].requires_grad = False

        self.torch_weights = nn.Parameter(torch.from_numpy(weights))
        node_counts = torch.as_tensor([h.n_nodes for h in self.trees])

        # Train the model
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr) 

        for epoch in range(self.epochs):
            mini_batches = create_mini_batches(X,Y, self.batch_size, True) 

            batch_cnt = 0
            loss_sum = 0
            accuracy_sum = 0
            n_estimators_sum = 0
            n_nodes_sum = 0

            with tqdm(total=X.shape[0], ncols=150, disable = not self.verbose) as pbar:
                for x,y in mini_batches:
                    # Prepare the target and apply all trees
                    pred = []
                    for i,e in enumerate(self.trees):
                        idx = e.apply(x)
                        pred.append(self.torch_weights[i] * self.torch_leafs[i][idx,:])
                    fbar = torch.stack(pred).mean(axis=0)

                    loss = self.compute_loss(fbar, pred, torch.tensor(y).long()) #+ 1e-1*(node_counts * self.torch_weights**2).sum()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # if epoch > 15: #burn in phase?
                    with torch.no_grad():
                        step_size = optimizer.param_groups[0]['lr']
                        proxed_w = prox(self.torch_weights.numpy(), self.ensemble_regularizer, self.normalize_weights, self.l_ensemble_reg, step_size, 1.0)
                        self.torch_weights.copy_(torch.from_numpy(proxed_w))

                    # compute some statistics 
                    loss_sum += loss.detach().numpy() #+ sum(abs(self.torch_weights.detach().numpy()))
                    accuracy_sum += accuracy_score(fbar.argmax(axis=1),y) * 100.0
                    n_estimators_sum += torch.count_nonzero(self.torch_weights)
                    n_nodes_sum += sum([0 if w == 0 else n for w,n in zip(self.torch_weights, node_counts)])

                    batch_cnt += 1 
                    pbar.update(x.shape[0])
                    
                    desc = '[{}/{}] loss {:2.4f} accuracy {:2.4f} n-est {:2.4f} n-nodes {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        loss_sum / batch_cnt,
                        accuracy_sum / batch_cnt, 
                        n_estimators_sum / batch_cnt,
                        n_nodes_sum / batch_cnt
                        # K_sum / batch_cnt
                    )
                    pbar.set_description(desc)

        trees = []
        weights = []
        for i in range(n_trees):
            w = self.torch_weights[i].detach().numpy()
            if w != 0:
                leafs = torch_leafs[i].detach().numpy()
                self.trees[i].model.tree_.value[:] = leafs[:,np.newaxis]
                trees.append(self.trees[i])
                weights.append(w)
        
        self.weights = weights
        self.trees = trees
        