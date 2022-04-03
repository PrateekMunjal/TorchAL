from matplotlib import use
import numpy as np
import torch
from statistics import mean

from al_utils import query_models
from .data import Data
import gc
import os
import math
import sys
from copy import deepcopy
import time
from scipy.spatial import distance_matrix
import pickle
import math
import torch.nn as nn
import torch.nn.functional as F
from pycls.models.coregcn import GCN
import torch.optim as optim
from .kcenter_greedy import kCenterGreedy
from tqdm import tqdm

import pycls.datasets.loader as imagenet_loader
from tqdm import tqdm
from typing import Tuple

from pycls.core.config import custom_dump_cfg


class EntropyLoss(nn.Module):
    """
    This class contains the entropy function implemented.
    """

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, applySoftMax=True):
        # Assuming x : [BatchSize, ]

        if applySoftMax:
            entropy = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        else:
            entropy = x * torch.log2(x)
        entropy = -1 * entropy.sum(dim=1)
        return entropy


class CoreSetMIPSampling:
    """
    Implements coreset MIP sampling operation
    """

    def __init__(self, cfg, dataObj, isMIP=False):
        self.dataObj = dataObj
        # self.gpu_list = args.device_ids
        self.cuda_id = torch.cuda.current_device()
        self.cfg = cfg
        self.isMIP = isMIP

    def dump_modified_cfg(self):
        """
        Modifies input configuration file (.yaml) to facilitate AL experiments.
        """

        temp_cfg = deepcopy(self.cfg)
        temp_cfg.ACTIVE_LEARNING.ACTIVATE = True
        temp_cfg.ACTIVE_LEARNING.LSET_PATH = os.path.join(temp_cfg.OUT_DIR, "lSet.npy")
        temp_cfg.ACTIVE_LEARNING.USET_PATH = os.path.join(temp_cfg.OUT_DIR, "uSet.npy")
        custom_dump_cfg(temp_cfg)

    @torch.no_grad()
    def get_representation(self, clf_model, idx_set, dataset):
        """Returns the representation (in our case activations from penultimate layer) for a given dataset.

        Args:
            clf_model (nn.Module): Image classifier.
            idx_set : Indexes specifying subset of the data to be covered.
            dataset : Reference to data.

        Returns:
            np.ndarray: Representations over the dataset.
        """

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            print(
                "Loading the model in data parallel where num_GPUS: {}".format(
                    self.cfg.NUM_GPUS
                )
            )
            clf_model = torch.nn.DataParallel(
                clf_model, device_ids=[i for i in range(self.cfg.NUM_GPUS)]
            )

            tempIdxSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=idx_set,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            tempIdxSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=idx_set,
                batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS),
                data=dataset,
            )
        features = []

        print(f"len(dataLoader): {len(tempIdxSetLoader)}")

        for i, (x, _) in enumerate(
            tqdm(tempIdxSetLoader, desc="Extracting Representations")
        ):
            # if i%50 == 0:
            #     print(f"Processed [{i}/{len(tempIdxSetLoader)}]")
            x = x.cuda(self.cuda_id)
            x = x.type(torch.cuda.FloatTensor)
            temp_z, _ = clf_model(x)
            features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features

    def gpu_compute_dists(self, M1, M2):
        """
        Computes L2 norm square on gpu
        Assume
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        # print(f"Function call to gpu_compute dists; M1: {M1.shape} and M2: {M2.shape}")
        M1_norm = (M1**2).sum(1).reshape(-1, 1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1, -1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def compute_dists(self, X, X_train):
        dists = (
            -2 * np.dot(X, X_train.T)
            + np.sum(X_train**2, axis=1)
            + np.sum(X**2, axis=1).reshape((-1, 1))
        )  # [:, np.newaxis]
        return dists

    def optimal_greedy_k_center(self, labeled, unlabeled):
        n_lSet = labeled.shape[0]
        lSetIds = np.arange(n_lSet)
        n_uSet = unlabeled.shape[0]
        uSetIds = n_lSet + np.arange(n_uSet)

        # order is important
        features = np.vstack((labeled, unlabeled))
        print(
            "Started computing distance matrix of {}x{}".format(
                features.shape[0], features.shape[0]
            )
        )
        start = time.time()
        distance_mat = self.compute_dists(features, features)
        end = time.time()
        print("Distance matrix computed in {} seconds".format(end - start))
        greedy_indices = []
        for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE):
            if i is not 0 and i % 500 == 0:
                print("Sampled {} samples".format(i))
            lab_temp_indexes = np.array(np.append(lSetIds, greedy_indices), dtype=int)
            # unlab_temp_indexes = np.array(list(set(uSetIds)-set(greedy_indices)))
            min_dist = np.min(distance_mat[lab_temp_indexes, n_lSet:], axis=0)
            active_index = np.argmax(min_dist)
            greedy_indices.append(n_lSet + active_index)

        remainSet = (
            set(np.arange(features.shape[0])) - set(greedy_indices) - set(lSetIds)
        )
        remainSet = np.array(list(remainSet))

        return greedy_indices - n_lSet, remainSet

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = [None for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)]
        greedy_indices_counter = 0

        # move cpu to gpu
        labeled = torch.from_numpy(labeled).cuda(0)
        unlabeled = torch.from_numpy(unlabeled).cuda(0)

        print(f"[GPU] Labeled.shape: {labeled.shape}")
        print(f"[GPU] Unlabeled.shape: {unlabeled.shape}")

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        st = time.time()

        min_dist, _ = torch.min(
            self.gpu_compute_dists(
                labeled[0, :].reshape((1, labeled.shape[1])), unlabeled
            ),
            dim=0,
        )
        min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

        print(f"time taken: {time.time() - st} seconds")
        temp_range = 500

        dist = np.empty((temp_range, unlabeled.shape[0]))
        for j in tqdm(
            range(1, labeled.shape[0], temp_range), desc="Getting first farthest index"
        ):
            if j + temp_range < labeled.shape[0]:
                dist = self.gpu_compute_dists(labeled[j : j + temp_range, :], unlabeled)

            else:

                dist = self.gpu_compute_dists(labeled[j:, :], unlabeled)

            min_dist = torch.cat(
                (min_dist, torch.min(dist, dim=0)[0].reshape((1, min_dist.shape[1])))
            )

            min_dist = torch.min(min_dist, dim=0)[0]
            min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        _, farthest = torch.max(min_dist, dim=1)
        # we do .item as farthest returned is of tensor type
        greedy_indices[greedy_indices_counter] = farthest.item()
        greedy_indices_counter += 1
        # greedy_indices.append(farthest)

        amount = self.cfg.ACTIVE_LEARNING.BUDGET_SIZE - 1

        for i in tqdm(range(amount), desc="Constructing Active set"):

            dist = self.gpu_compute_dists(
                unlabeled[greedy_indices[greedy_indices_counter - 1], :].reshape(
                    (1, unlabeled.shape[1])
                ),
                unlabeled,
            )

            min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))

            min_dist, _ = torch.min(min_dist, dim=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

            _, farthest = torch.max(min_dist, dim=1)

            greedy_indices[greedy_indices_counter] = farthest.item()
            greedy_indices_counter += 1

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))

        self.dump_modified_cfg()

        if self.isMIP:
            return greedy_indices, remainSet, math.sqrt(np.max(min_dist))
        else:
            return greedy_indices, remainSet

    def query(self, lSet, uSet, clf_model, dataset):

        assert (
            clf_model.training == False
        ), "Classification model expected in training mode"
        assert (
            clf_model.penultimate_active == True
        ), "Classification model is expected in penultimate mode"

        print("Extracting Lset Representations")
        lb_repr = self.get_representation(
            clf_model=clf_model, idx_set=lSet, dataset=dataset
        )
        print("Extracting Uset Representations")
        ul_repr = self.get_representation(
            clf_model=clf_model, idx_set=uSet, dataset=dataset
        )

        print("lb_repr.shape: ", lb_repr.shape)
        print("ul_repr.shape: ", ul_repr.shape)

        if self.isMIP == True:
            features = np.vstack((lb_repr, ul_repr))
            print("features.shape: ", features.shape)
            pickle.dump(features, open("feature_vectors_pickled", "wb"))
            print("Features written successfully")
            greedy_indexes, remainSet, delta_ub = self.greedy_k_center(
                labeled=lb_repr, unlabeled=ul_repr
            )
            print("delta_ub : ", delta_ub)
            activeSet = uSet[greedy_indexes]
            remainSet = uSet[remainSet]
            # Don't know why I am saving but it might be needed for gurobi solution
            np.save("greedy_indexes.npy", activeSet)
            print("Exiting as this method is not completely functional yet")
            sys.exit(0)

        else:
            print("Solving K Center Greedy Approach")
            start = time.time()
            greedy_indexes, remainSet = self.greedy_k_center(
                labeled=lb_repr, unlabeled=ul_repr
            )
            # greedy_indexes, remainSet = self.optimal_greedy_k_center(labeled=lb_repr, unlabeled=ul_repr)
            end = time.time()
            print("Time taken to solve K center: {} seconds".format(end - start))
            activeSet = uSet[greedy_indexes]
            remainSet = uSet[remainSet]

        return activeSet, remainSet


class Sampling:
    """
    Here we implement different sampling methods which are used to sample
    active learning points from unlabelled set.
    """

    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.cuda_id = (
            0
            if cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("ensemble")
            else torch.cuda.current_device()
        )
        self.dataObj = dataObj

    def gpu_compute_dists(self, M1, M2):
        """
        Computes L2 norm square on gpu
        Assume
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1, 1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1, -1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def get_predictions(self, clf_model, idx_set, dataset):
        """Returns the prob. predictions on dataset using input classifier.

        Args:
            clf_model (nn.Module): Image classifier
            idx_set : Indexes specifying subset of the data to be covered.
            dataset : Reference to data.

        Returns:
            np.ndarray: Prob. predicitions.
        """
        # Used by bald acquisition
        if self.cfg.TRAIN.DATASET == "IMAGENET":
            tempIdxSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=idx_set,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            tempIdxSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=idx_set,
                batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS),
                data=dataset,
            )
        preds = []
        for i, (x, _) in enumerate(
            tqdm(
                tempIdxSetLoader,
                desc="Collecting predictions in get_predictions function",
            )
        ):
            x = x.cuda(self.cuda_id)
            x = x.type(torch.cuda.FloatTensor)

            temp_pred = clf_model(x)

            # To get probabilities
            temp_pred = torch.nn.functional.softmax(temp_pred, dim=1)
            preds.append(temp_pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        return preds

    def random(self, uSet, budgetSize):
        """
        Chooses <budgetSize> number of data points randomly from uSet.

        NOTE: The returned uSet is modified such that it does not contain active datapoints.

        INPUT
        ------

        uSet: np.ndarray, It describes the index set of unlabelled set.

        budgetSize: int, The number of active data points to be chosen for active learning.

        OUTPUT
        -------

        Returns activeSet, uSet
        """

        np.random.seed(self.cfg.RNG_SEED)

        assert isinstance(
            uSet, np.ndarray
        ), "Expected uSet of type np.ndarray whereas provided is dtype:{}".format(
            type(uSet)
        )
        assert isinstance(
            budgetSize, int
        ), "Expected budgetSize of type int whereas provided is dtype:{}".format(
            type(budgetSize)
        )
        assert budgetSize > 0, "Expected a positive budgetSize"
        assert budgetSize < len(
            uSet
        ), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}".format(
            len(uSet), budgetSize
        )

        tempIdx = [i for i in range(len(uSet))]
        np.random.shuffle(tempIdx)
        activeSet = uSet[tempIdx[0:budgetSize]]
        uSet = uSet[tempIdx[budgetSize:]]

        temp_cfg = deepcopy(self.cfg)

        # Write cfg file
        self.dump_modified_cfg()

        return activeSet, uSet

    def bald(self, budgetSize, uSet, clf_model, dataset):
        "Implements BALD acquisition function where we maximize information gain."

        clf_model.cuda(self.cuda_id)

        assert (
            self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0
        ), "Expected dropout iterations > 0."

        # Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            uSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=uSet,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            uSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=uSet,
                batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS),
                data=dataset,
            )

        n_uPts = len(uSet)
        # Source Code was in tensorflow
        # To provide same readability we use same variable names where ever possible
        # Original TF-Code: https://github.com/Riashat/Deep-Bayesian-Active-Learning/blob/master/MC_Dropout_Keras/Dropout_Bald_Q10_N1000_Paper.py#L223

        # Heuristic: G_X - F_X
        score_All = np.zeros(shape=(n_uPts, self.cfg.MODEL.NUM_CLASSES))
        all_entropy_dropout = np.zeros(shape=(n_uPts))

        for d in tqdm(
            range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS),
            desc="Dropout Iterations",
        ):
            dropout_score = self.get_predictions(
                clf_model=clf_model, idx_set=uSet, dataset=dataset
            )

            score_All += dropout_score

            # computing F_x
            dropout_score_log = np.log2(
                dropout_score + 1e-6
            )  # Add 1e-6 to avoid log(0)
            Entropy_Compute = -np.multiply(dropout_score, dropout_score_log)
            Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)

            all_entropy_dropout += Entropy_per_Dropout

        Avg_Pi = np.divide(score_All, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        Log_Avg_Pi = np.log2(Avg_Pi + 1e-6)
        Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
        G_X = Entropy_Average_Pi
        Average_Entropy = np.divide(
            all_entropy_dropout, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS
        )
        F_X = Average_Entropy

        U_X = G_X - F_X
        print("U_X.shape: ", U_X.shape)

        # To manually inspect if sampling works correctly
        scores_save_path = self.cfg.OUT_DIR
        os.makedirs(scores_save_path, exist_ok=True)  # just to be safe
        with open(os.path.join(scores_save_path, "actualScores.txt"), "w") as fpw:
            for temp_idx, temp_rank in zip(uSet, U_X):
                fpw.write(f"{temp_idx}\t{temp_rank:.6f}\n")

        fpw.close()

        sorted_idx = np.argsort(U_X)[
            ::-1
        ]  # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        # Setting task model in train mode for further learning
        clf_model.train()

        # Write cfg file
        self.dump_modified_cfg()

        return activeSet, remainSet

    def ensemble_bald(self, budgetSize, uSet, clf_models, dataset):
        "Implements BALD acquisition function where we maximize information gain."

        for cmodel in clf_models:
            cmodel.cuda(self.cuda_id)
            cmodel.eval()

        assert len(clf_models) != 0, "Expected number of classification models > 0."

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            uSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=uSet,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            uSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=uSet,
                batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS),
                data=dataset,
            )

        n_uPts = len(uSet)
        # Source Code was in tensorflow
        # To provide same readability we use same variable names where ever possible
        # Original TF-Code: https://github.com/Riashat/Deep-Bayesian-Active-Learning/blob/master/MC_Dropout_Keras/Dropout_Bald_Q10_N1000_Paper.py#L223

        # Heuristic: G_X - F_X
        score_All = np.zeros(shape=(n_uPts, self.cfg.MODEL.NUM_CLASSES))
        all_entropy_dropout = np.zeros(shape=(n_uPts))

        for d in tqdm(
            range(len(clf_models)),
            desc="Dropout Iterations via Fwd pass thru ensembles",
        ):
            # print("Dropout iteration: {}".format(d))
            dropout_score = self.get_predictions(
                clf_model=clf_models[d], idx_set=uSet, dataset=dataset
            )
            # print("dropout_score.shape: ",dropout_score.shape)

            score_All += dropout_score

            # computing F_x
            dropout_score_log = np.log2(dropout_score + 1e-6)
            Entropy_Compute = -np.multiply(dropout_score, dropout_score_log)
            Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)

            all_entropy_dropout += Entropy_per_Dropout

        Avg_Pi = np.divide(score_All, len(clf_models))
        Log_Avg_Pi = np.log2(Avg_Pi + 1e-6)
        Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
        G_X = Entropy_Average_Pi
        Average_Entropy = np.divide(all_entropy_dropout, len(clf_models))
        F_X = Average_Entropy

        U_X = G_X - F_X
        print("U_X.shape: ", U_X.shape)
        sorted_idx = np.argsort(U_X)[
            ::-1
        ]  # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        # Setting task model in train mode for further learning
        for cmodel in clf_models:
            cmodel.train()
        return activeSet, remainSet

    def dbal(self, budgetSize, uSet, clf_model, dataset):
        """
        Implements deep bayesian active learning where uncertainty is measured by
        maximizing entropy of predictions. This uncertainty method is choosen following
        the recent state of the art approach, VAAL. [SOURCE: Implementation Details in VAAL paper]

        In bayesian view, predictions are computed with the help of dropouts and
        Monte Carlo approximation
        """
        clf_model.cuda(self.cuda_id)

        # Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            # print("True")
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        # clf_model = torch.nn.DataParallel(clf_model, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        assert (
            self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0
        ), "Expected dropout iterations > 0."

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            uSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=uSet,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            uSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=uSet,
                batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS),
                data=dataset,
            )

        u_scores = []
        n_uPts = len(uSet)
        ptsProcessed = 0

        entropy_loss = EntropyLoss()

        print("len usetLoader: {}".format(len(uSetLoader)))
        temp_i = 0

        for k, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Feed Forward")):
            temp_i += 1

            x_u = x_u.type(torch.cuda.FloatTensor)
            z_op = np.zeros((x_u.shape[0], self.cfg.MODEL.NUM_CLASSES), dtype=float)
            for i in range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS):
                x_u = x_u.cuda(self.cuda_id)
                temp_op = clf_model(x_u)
                # Till here z_op represents logits of p(y|x).
                # So to get probabilities
                temp_op = torch.nn.functional.softmax(temp_op, dim=1)
                z_op = np.add(z_op, temp_op.cpu().numpy())

            z_op /= self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS

            z_op = torch.from_numpy(z_op).cuda(self.cuda_id)
            entropy_z_op = entropy_loss(z_op, applySoftMax=False)

            # Now entropy_z_op = Sum over all classes{ -p(y=c|x) log p(y=c|x)}
            u_scores.append(entropy_z_op.cpu().numpy())
            ptsProcessed += x_u.shape[0]

        u_scores = np.concatenate(u_scores, axis=0)

        # To manually inspect if sampling works correctly
        scores_save_path = self.cfg.OUT_DIR
        os.makedirs(scores_save_path, exist_ok=True)  # just to be safe
        with open(os.path.join(scores_save_path, "actualScores.txt"), "w") as fpw:
            for temp_idx, temp_rank in zip(uSet, u_scores):
                fpw.write(f"{temp_idx}\t{temp_rank:.6f}\n")

        fpw.close()

        sorted_idx = np.argsort(u_scores)[
            ::-1
        ]  # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]

        # Write cfg file
        self.dump_modified_cfg()

        return activeSet, remainSet

    def ensemble_var_R(self, budgetSize, uSet, clf_models, dataset):
        """
        Implements ensemble variance_ratio measured as the number of disagreement in committee
        with respect to the predicted class.
        If f_m is number of members agreeing to predicted class then
        variance ratio(var_r) is evaludated as follows:

            var_r = 1 - (f_m / T); where T is number of commitee members

        For more details refer equation 4 in
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf
        """
        from scipy import stats

        T = len(clf_models)

        for cmodel in clf_models:
            cmodel.cuda(self.cuda_id)
            cmodel.eval()

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            uSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=uSet,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            uSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=uSet,
                batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS),
                data=dataset,
            )

        print("len usetLoader: {}".format(len(uSetLoader)))

        temp_i = 0
        var_r_scores = np.zeros((len(uSet), 1), dtype=float)

        for k, (x_u, _) in enumerate(
            tqdm(uSetLoader, desc="uSet Forward Passes through " + str(T) + " models")
        ):

            x_u = x_u.type(torch.cuda.FloatTensor)
            ens_preds = np.zeros((x_u.shape[0], T), dtype=float)
            for i in range(len(clf_models)):
                x_u = x_u.cuda(self.cuda_id)
                temp_op = clf_models[i](x_u)
                _, temp_pred = torch.max(temp_op, 1)
                temp_pred = temp_pred.cpu().numpy()
                ens_preds[:, i] = temp_pred
                # temp_op = temp_op.cpu().numpy()

            _, mode_cnt = stats.mode(ens_preds, 1)
            temp_varr = 1.0 - (mode_cnt / T * 1.0)
            var_r_scores[temp_i : temp_i + x_u.shape[0]] = temp_varr
            # z_op = np.add(z_op, temp_op.cpu().numpy())

            temp_i = temp_i + x_u.shape[0]

        var_r_scores = np.squeeze(np.array(var_r_scores))

        print("var_r_scores.shape: {}".format(var_r_scores.shape))
        # To manually inspect if sampling works correctly
        scores_save_path = self.cfg.OUT_DIR
        os.makedirs(scores_save_path, exist_ok=True)  # just to be safe
        with open(os.path.join(scores_save_path, "actualScores.txt"), "w") as fpw:
            for temp_idx, temp_rank in zip(uSet, var_r_scores):
                fpw.write(f"{temp_idx}\t{temp_rank:.6f}\n")

        fpw.close()

        sorted_idx = np.argsort(var_r_scores)[
            ::-1
        ]  # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        return activeSet, remainSet

    def ensemble_dbal(self, budgetSize, uSet, clf_models, dataset):
        """
        Implements ENSEMBLE using deep bayesian active learning where uncertainty is measured by
        maximizing entropy of predictions. This uncertainty method is choosen following
        the recent state of the art approach, VAAL. [SOURCE: Implementation Details in VAAL paper]

        In bayesian view, predictions are computed with the help of dropouts and
        Monte Carlo approximation
        """
        for cmodel in clf_models:
            cmodel.cuda(self.cuda_id)
            cmodel.eval()

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            uSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=uSet,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            uSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=uSet,
                batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS),
                data=dataset,
            )

        u_scores = []
        n_uPts = len(uSet)
        ptsProcessed = 0

        entropy_loss = EntropyLoss()

        print("len usetLoader: {}".format(len(uSetLoader)))
        temp_i = 0
        for k, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Feed Forward")):
            temp_i += 1

            x_u = x_u.type(torch.cuda.FloatTensor)
            z_op = np.zeros((x_u.shape[0], self.cfg.MODEL.NUM_CLASSES), dtype=float)
            for i in range(len(clf_models)):
                x_u = x_u.cuda(self.cuda_id)
                temp_op = clf_models[i](x_u)
                # Till here z_op represents logits of p(y|x).
                # So to get probabilities
                temp_op = torch.nn.functional.softmax(temp_op, dim=1)
                z_op = np.add(z_op, temp_op.cpu().numpy())

            z_op /= len(clf_models)

            z_op = torch.from_numpy(z_op).cuda(self.cuda_id)

            entropy_z_op = entropy_loss(z_op, applySoftMax=False)
            # Now entropy_z_op = Sum over all classes{ -p(y=c|x) log p(y=c|x)}
            u_scores.append(entropy_z_op.cpu().numpy())
            ptsProcessed += x_u.shape[0]

        u_scores = np.concatenate(u_scores, axis=0)
        # print("uscores shape: ",u_scores.shape)
        sorted_idx = np.argsort(u_scores)[
            ::-1
        ]  # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        return activeSet, remainSet

    def centre_of_gravity(self, budgetSize, lSet, uSet, model, dataset, istopK=True):
        num_classes = self.cfg.MODEL.NUM_CLASSES
        """
        Implements the center of gravity as a acquisition function. The uncertainty is measured as 
        euclidean distance of data point form center of gravity. The uncertainty increases with eucliden distance.

        INPUT
        ------

        uSet: np.ndarray, It describes the index set of unlabelled set.

        budgetSize: int, The number of active data points to be chosen for active learning.

        OUTPUT
        -------

        Returns activeSet, uSet   
        """

        assert (
            model.training == False
        ), "Model expected in eval mode whereas currently it is in {}".format(
            model.training
        )
        assert len(lSet) != 0, "lSet cannot be empty."
        assert len(uSet) != 0, "uSet cannot be empty."

        clf = model
        clf.cuda(self.cuda_id)

        luSet = np.append(lSet, uSet)

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            luSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=luSet,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            luSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=luSet,
                batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS),
                data=dataset,
            )

        z_points = []

        for i, (x_u, _) in enumerate(tqdm(luSetLoader, desc="luSet Activations")):
            x_u = x_u.cuda(self.cuda_id)
            x_u = x_u.type(torch.cuda.FloatTensor)
            temp_z, _ = clf(x_u)
            z_points.append(temp_z.cpu().numpy())

        z_points = np.concatenate(z_points, axis=0)

        # Segregate labeled and unlabeled activations
        # As we use sequential data loader therefore lSet activations are present first followed by uSet activations
        l_acts = z_points[: len(lSet)]
        u_acts = z_points[len(lSet) :]

        print(f"u_latent_acts.shape: {u_acts.shape}")
        print(f"l_latent_acts.shape: {l_acts.shape}")

        cog = np.mean(z_points, axis=0)
        cog = torch.from_numpy(cog).cuda(self.cuda_id)
        cog = cog.reshape([1, cog.shape[0]])

        dist = [100000.0 for i in range(len(uSet))]
        dist_idx = 0

        u_acts = torch.from_numpy(u_acts).cuda(self.cuda_id)
        temp_bs = self.cfg.TRAIN.BATCH_SIZE

        for i in tqdm(
            range(0, u_acts.shape[0], temp_bs), desc="Computing Distance matrix"
        ):
            end_index = (
                i + temp_bs if i + temp_bs < u_acts.shape[0] else u_acts.shape[0]
            )  # to avoid out of index access
            z_u = u_acts[i:end_index, :]
            dist[i:end_index] = torch.sqrt((cog - z_u).pow(2).sum(1)).cpu().numpy()

            dist_idx = end_index

        assert dist_idx == len(
            uSet
        ), "dist_idx is expected to be {} whereas it is {} and len(uSet): {}".format(
            len(uSet), dist_idx, len(uSet)
        )
        # Now u_ranks has shape: [U_Size x 1]

        dist = np.array(dist)

        print("dist.shape: {}".format(dist.shape))
        # To manually inspect if sampling works correctly
        scores_save_path = self.cfg.OUT_DIR
        os.makedirs(scores_save_path, exist_ok=True)  # just to be safe
        with open(os.path.join(scores_save_path, "actualScores.txt"), "w") as fpw:
            for temp_idx, temp_rank in zip(uSet, dist):
                fpw.write(f"{temp_idx}\t{temp_rank:.6f}\n")

        fpw.close()

        # Argsort - returns the indices such that elements with large values come in the last
        # For COG: the more a point is distant from cog, the more we are uncertain about it
        # so to get the indices for which corresponding values are maximum -> we reverse the result of argsort using [::-1]
        sorted_idx = np.argsort(dist)[::-1]
        print("Sorting done..")
        if istopK:
            print("---COG [Topk] Activated---")
            activeSet = uSet[sorted_idx[0:budgetSize]]
            remainSet = uSet[sorted_idx[budgetSize:]]

            # Write cfg file
            self.dump_modified_cfg()
            return activeSet, remainSet

        print("---COG [Uniform Binning] Activated]---")
        # index of u_ranks serve as key to refer in u_idx
        # we add -1 for reversing the sorted array
        activeSet, remainSet = self.bin_uniformly(
            sorted_idx=sorted_idx, budgetSize=budgetSize
        )

        activeSet = uSet[activeSet]
        remainSet = uSet[remainSet]

        self.dump_modified_cfg()
        return activeSet, remainSet

    def uncertainty_mix(
        self, budgetSize: int, lSet: np.ndarray, uSet: np.ndarray, model, dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Implements the uncertainty principle as a acquisition function.

        Args:
            budgetSize (int): [description]
            lSet (np.ndarray): [description]
            uSet (np.ndarray): [description]
            model ([type]): [description]
            dataset ([type]): [description]

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert (
            model.training == False
        ), "Model expected in eval mode whereas currently it is in {}".format(
            model.training
        )

        assert isinstance(model, nn.Module)
        assert isinstance(
            dataset, torch.utils.data.Dataset
        ), f"Datatype Mismatch. Expected `torch.utils.data.Dataset` but got {type(dataset)}"
        assert (
            model.training == False
        ), "Expected the model to be in eval mode but it is in train mode as model.training give True"

        clf = model

        u_ranks = []

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            print(
                "Loading the model in data parallel where num_GPUS: {}".format(
                    self.cfg.NUM_GPUS
                )
            )
            clf = torch.nn.DataParallel(
                clf, device_ids=[i for i in range(self.cfg.NUM_GPUS)]
            )
            uSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=uSet,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            uSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset
            )

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            x_u = x_u.cuda(0)

            temp_u_rank = clf(x_u)
            # get probs
            temp_u_rank = torch.nn.functional.softmax(temp_u_rank, dim=1)
            # find the max prob for having certainty of a clf
            temp_u_rank, _ = torch.max(temp_u_rank, dim=1)
            u_ranks.append(temp_u_rank.cpu().numpy())

        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")

        # To manually inspect if sampling works correctly
        scores_save_path = self.cfg.OUT_DIR
        os.makedirs(scores_save_path, exist_ok=True)  # just to be safe
        with open(os.path.join(scores_save_path, "actualScores.txt"), "w") as fpw:
            for temp_idx, temp_rank in zip(uSet, u_ranks):
                fpw.write(f"{temp_idx}\t{temp_rank:.6f}\n")

        fpw.close()

        sorted_idx = np.argsort(
            u_ranks
        )  # argsort helps to return the indices of u_ranks such that indices with large values comes at last

        firsthalfbudget = budgetSize // 2
        secondhalfbudget = budgetSize - firsthalfbudget

        sorted_firsthalf = sorted_idx[:firsthalfbudget]
        sorted_secondhalf = sorted_idx[::-1][:secondhalfbudget]
        # activeSet = sorted_idx[:budgetSize]
        activeSet = np.append(sorted_firsthalf, sorted_secondhalf)
        remainSetIdx = np.array(list(set(sorted_idx).difference(set(activeSet))))
        activeSet = uSet[activeSet]

        # remainSet = uSet[sorted_idx[budgetSize:]]
        remainSet = uSet[remainSetIdx]

        temp_cfg = deepcopy(self.cfg)

        # Write cfg file
        self.dump_modified_cfg()

        return activeSet, remainSet

    @torch.no_grad()
    def uncertainty(
        self, budgetSize: int, lSet: np.ndarray, uSet: np.ndarray, model, dataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Implements the uncertainty principle as a acquisition function.

        Args:
            budgetSize (int): [description]
            lSet (np.ndarray): [description]
            uSet (np.ndarray): [description]
            model ([type]): [description]
            dataset ([type]): [description]

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert (
            model.training == False
        ), "Model expected in eval mode whereas currently it is in {}".format(
            model.training
        )

        assert isinstance(model, nn.Module)
        assert isinstance(
            dataset, torch.utils.data.Dataset
        ), f"Datatype Mismatch. Expected `torch.utils.data.Dataset` but got {type(dataset)}"
        assert (
            model.training == False
        ), "Expected the model to be in eval mode but it is in train mode as model.training give True"

        clf = model

        u_ranks = []

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            print(
                "Loading the model in data parallel where num_GPUS: {}".format(
                    self.cfg.NUM_GPUS
                )
            )
            clf = torch.nn.DataParallel(
                clf, device_ids=[i for i in range(self.cfg.NUM_GPUS)]
            )
            uSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=uSet,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            uSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset
            )

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            x_u = x_u.cuda(0)

            temp_u_rank = clf(x_u)
            # get probs
            temp_u_rank = torch.nn.functional.softmax(temp_u_rank, dim=1)
            # find the max prob for having certainty of a clf
            temp_u_rank, _ = torch.max(temp_u_rank, dim=1)
            u_ranks.append(temp_u_rank.cpu().numpy())

        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")

        # To manually inspect if sampling works correctly
        scores_save_path = self.cfg.OUT_DIR
        os.makedirs(scores_save_path, exist_ok=True)  # just to be safe
        with open(os.path.join(scores_save_path, "actualScores.txt"), "w") as fpw:
            for temp_idx, temp_rank in zip(uSet, u_ranks):
                fpw.write(f"{temp_idx}\t{temp_rank:.6f}\n")

        fpw.close()

        sorted_idx = np.argsort(
            u_ranks
        )  # argsort helps to return the indices of u_ranks such that indices with large values comes at last
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]

        temp_cfg = deepcopy(self.cfg)

        # Write cfg file
        self.dump_modified_cfg()

        return activeSet, remainSet

    def dump_modified_cfg(self):

        temp_cfg = deepcopy(self.cfg)
        temp_cfg.ACTIVE_LEARNING.ACTIVATE = True
        temp_cfg.ACTIVE_LEARNING.LSET_PATH = os.path.join(temp_cfg.OUT_DIR, "lSet.npy")
        temp_cfg.ACTIVE_LEARNING.USET_PATH = os.path.join(temp_cfg.OUT_DIR, "uSet.npy")
        custom_dump_cfg(temp_cfg)

    def bin_uniformly(self, sorted_idx, budgetSize):
        """
        Bins sorted_idx uniformly. Then forms active set of <budgetSize> size where data points
        are sampled uniformly from bins i.e from each bin equal number of points are probable in
        activeSet.

        INPUT
        ------

        sorted_idx: np.ndarray, array of sorted indexes set of unlabelled set

        budgetSize: int, size of activeSet

        OUTPUT
        -------
        On successful completion it returns activeSet, remainingSet where remainingSet serves as uSet
        for further iterations.
        """
        n_bins = self.cfg.ACTIVE_LEARNING.N_BINS
        bin_size = int(len(sorted_idx) / n_bins)
        items_per_bin = int(math.ceil(budgetSize / n_bins))
        activeSet = None

        sorted_idxSet = set(sorted_idx)
        lowIdxBin = 0
        highIdxBin = 0

        for i in tqdm(range(n_bins), desc="Binning Uniformly"):
            lowIdxBin = i * bin_size
            highIdxBin = lowIdxBin + bin_size - 1
            # highIdxBin+1 will be low for next batch
            if highIdxBin + bin_size > len(sorted_idx) - 1:
                highIdxBin = len(sorted_idx) - 1

            # print("[low,high]: [{},{}]".format(lowIdxBin,highIdxBin))

            bin_interval = sorted_idx[lowIdxBin : highIdxBin + 1]

            if activeSet is not None:
                assert (
                    len(set(activeSet) & set(bin_interval)) == 0
                ), "Unexpected intersection between bin_interval\
                    and activeSet."

            if activeSet is not None and len(activeSet) + items_per_bin > budgetSize:
                items_per_bin = budgetSize - len(activeSet)

            tempActive = np.random.choice(
                a=bin_interval, size=items_per_bin, replace=False
            )
            if activeSet is None:
                activeSet = tempActive
            else:
                # intersection should be empty
                assert (
                    len(set(activeSet) & set(tempActive)) == 0
                ), "There is intersection between activeSet and tempActive. The intersection is {}".format(
                    set(activeSet) & set(tempActive)
                )
                activeSet = np.append(activeSet, tempActive)

            if len(activeSet) == budgetSize:
                break

        assert (
            len(activeSet) == budgetSize
        ), "Something wrong in implementation. Program expects the length of activeSet to be same as\
            desired budgetSize. Currently budgetSize: {} and len(ActiveSet):{}".format(
            budgetSize, len(activeSet)
        )

        activeSet = set(activeSet)
        remainSet = sorted_idxSet - activeSet

        # converting set to np.ndarrays
        activeSet = np.array(list(activeSet))
        remainSet = np.array(list(remainSet))

        return activeSet, remainSet

    def uncertainty_uniform_discretize(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function. However, the active samples
        are chosen by binning at uniform intervals and then choosing.

        INPUT
        ------
        lSet: np.ndarray, It describes the index set of labelled set.

        uSet: np.ndarray, It describes the index set of unlabelled set.

        budgetSize: int, The number of active data points to be chosen for active learning.

        dataset: reference to Dataset on which model is trained

        model: classfication model

        OUTPUT
        -------

        Returns activeSet, uSet

        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert (
            model.training == False
        ), "Model expected in eval mode whereas currently it is in {}".format(
            model.training
        )

        clf = model

        # uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=self.cfg.TRAIN.BATCH_SIZE,data=dataset)

        if self.cfg.TRAIN.DATASET == "IMAGENET":
            uSetLoader = imagenet_loader.construct_loader_no_aug(
                cfg=self.cfg,
                indices=uSet,
                isDistributed=False,
                isShuffle=False,
                isVaalSampling=False,
            )
        else:
            uSetLoader = self.dataObj.getSequentialDataLoader(
                indexes=uSet, batch_size=self.cfg.TRAIN.BATCH_SIZE, data=dataset
            )

        u_ranks = []
        for i, (x_u, _) in enumerate(uSetLoader):
            x_u = x_u.type(torch.cuda.FloatTensor)
            # x_u = x_u.cuda(self.cuda_id)

            temp_u_rank = clf(x_u)
            temp_u_rank, _ = torch.max(temp_u_rank, dim=1)
            u_ranks.append(temp_u_rank.cpu().numpy())

        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[
            ::-1
        ]  # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.

        activeSet, remainSet = self.bin_uniformly(
            sorted_idx=sorted_idx, budgetSize=budgetSize
        )

        activeSet = uSet[activeSet]
        remainSet = uSet[remainSet]
        return activeSet, remainSet

    # def core_gcn(self, budgetSize:int, lSet:np.ndarray, uSet:np.ndarray, model, dataset)->Tuple[np.ndarray, np.ndarray]:
    #     """Implements the coreGCN AL method.

    #     Args:
    #         budgetSize (int): [description]
    #         lSet (np.ndarray): [description]
    #         uSet (np.ndarray): [description]
    #         model ([type]): [description]
    #         dataset ([type]): [description]

    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: [description]
    #     """

    #     def aff_to_adj(x, y=None):
    #         x = x.detach().cpu().numpy()
    #         adj = np.matmul(x, x.transpose())
    #         adj +=  -1.0*np.eye(adj.shape[0])
    #         adj_diag = np.sum(adj, axis=0) #rowise sum
    #         adj = np.matmul(adj, np.diag(1/adj_diag))
    #         adj = adj + np.eye(adj.shape[0])
    #         adj = torch.Tensor(adj).cuda()

    #         return adj

    #     def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    #         lnl = torch.log(scores[lbl])
    #         lnu = torch.log(1 - scores[nlbl])
    #         labeled_score = torch.mean(lnl)
    #         unlabeled_score = torch.mean(lnu)
    #         bce_adj_loss = -labeled_score - l_adj*unlabeled_score

    #         return bce_adj_loss

    #     num_classes = self.cfg.MODEL.NUM_CLASSES
    #     assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

    #     assert isinstance(model, nn.Module)
    #     assert isinstance(dataset, torch.utils.data.Dataset),f'Datatype Mismatch. Expected `torch.utils.data.Dataset` but got {type(dataset)}'
    #     assert model.training == False, "Expected the model to be in eval mode but it is in train mode as model.training give True"

    #     clf = model
    #     clf.cuda(self.cuda_id)

    #     ulSet = np.append(uSet, lSet)
    #     binary_labels = torch.cat((torch.zeros([len(uSet), 1]),(torch.ones([len(lSet),1]))),0)

    #     if self.cfg.TRAIN.DATASET == "IMAGENET":
    #         ulSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=ulSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
    #     else:
    #         ulSetLoader = self.dataObj.getSequentialDataLoader(indexes=ulSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)

    #     # z_points = []
    #     features = torch.tensor([]).cuda(self.cuda_id)

    #     oldmode_penultimate = clf.penultimate_active
    #     clf.penultimate_active = True

    #     with torch.no_grad():
    #         for i, (x_u, _) in enumerate(tqdm(ulSetLoader, desc="ulSet Activations")):
    #             x_u = x_u.cuda(self.cuda_id)
    #             x_u = x_u.type(torch.cuda.FloatTensor)
    #             # with torch.no_grad():
    #             temp_z, _ = clf(x_u)
    #             # z_points.append(temp_z.cpu().numpy())
    #             features = torch.cat((features, temp_z), 0)

    #     # features = np.concatenate(z_points, axis=0)
    #     clf.penultimate_active = oldmode_penultimate

    #     print('Normalizing features....')
    #     features = nn.functional.normalize(features) #(torch.tensor(features).cuda(self.cuda_id))
    #     print('Computing Adjacency matrix...')
    #     adj = aff_to_adj(features.clone())

    #     print('Building GCN...')
    #     print(f'GCN config: nFeat={features.shape[1]} nhid={128} dropout={0.3}')
    #     gcn_module = GCN(nfeat=features.shape[1],
    #                      nhid=128,
    #                      nclass=1,
    #                      dropout=0.3).cuda(self.cuda_id)

    #     gcn_module.train()

    #     gcn_optim = optim.Adam(gcn_module.parameters(), lr=1e-3, weight_decay=5e-4)

    #     # lbl = np.arange(0, len(lSet), 1)
    #     # nlbl = np.arange(len(lSet), len(uSet), 1)

    #     lbl = np.arange(len(uSet), len(ulSet), 1)
    #     nlbl = np.arange(0, len(uSet), 1)

    #     # lbl = torch.tensor(lbl).cuda(self.cuda_id)
    #     # nlbl = torch.tensor(nlbl).cuda(self.cuda_id)

    #     print("features.shape: ",features.shape)
    #     print("adj.shape: ",adj.shape)

    #      ############
    #     for gcn_epoch in tqdm(range(200), desc="Training GCN"):

    #         gcn_optim.zero_grad()
    #         outputs, _, _ = gcn_module(features, adj)
    #         # print('~~ for loop: outputs.requires_grad: ', outputs.requires_grad, 'outputs.shape: ', outputs.shape)
    #         lamda = 1.2
    #         loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
    #         loss.backward()
    #         gcn_optim.step()

    #         if gcn_epoch%20 == 0:
    #             print(f'GCN Training Loss: {loss.item()} at epoch: {gcn_epoch}')

    #     gcn_module.eval()

    #     with torch.no_grad():
    #         inputs = features.cuda(self.cuda_id)
    #         labels = binary_labels.cuda(self.cuda_id)
    #         scores, _, feat = gcn_module(inputs, adj)

    #     # if method == "CoreGCN":
    #     feat = feat.detach().cpu().numpy()
    #     new_av_idx = np.arange(len(uSet),len(ulSet))

    #     sampling2 = kCenterGreedy(feat)
    #     batch2 = sampling2.select_batch_(new_av_idx, budgetSize) # indices of the active set
    #     other_idx = [x for x in range(len(uSet)) if x not in batch2]
    #     # arg = other_idx + batch2

    #     # sorted_idx = np.argsort(u_ranks) #argsort helps to return the indices of u_ranks such that indices with large values comes at last
    #     # activeSet = sorted_idx[:budgetSize]

    #     activeSet = batch2

    #     activeSet = uSet[activeSet]
    #     # remainSet = uSet[sorted_idx[budgetSize:]]
    #     remainSet = uSet[other_idx]

    #     temp_cfg = deepcopy(self.cfg)

    #     # Write cfg file
    #     self.dump_modified_cfg()

    #     return activeSet, remainSet

    # def tod(self, budgetSize:int, lSet:np.ndarray, uSet:np.ndarray, model, dataset)->Tuple[np.ndarray, np.ndarray]:
    #     """Implements the TOD principle as a acquisition function.

    #     Args:
    #         budgetSize (int): [description]
    #         lSet (np.ndarray): [description]
    #         uSet (np.ndarray): [description]
    #         model ([type]): [description]
    #         dataset ([type]): [description]

    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: [description]
    #     """
    #     num_classes = self.cfg.MODEL.NUM_CLASSES
    #     assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

    #     assert isinstance(model, nn.Module)
    #     assert isinstance(dataset, torch.utils.data.Dataset),f'Datatype Mismatch. Expected `torch.utils.data.Dataset` but got {type(dataset)}'
    #     assert model.training == False, "Expected the model to be in eval mode but it is in train mode as model.training give True"

    #     clf = model

    #     print('Before Loading model: \n config')
    #     print(self.cfg.ACTIVE_LEARNING.AL_ITERATION)
    #     print('prev ckpt method: ', self.cfg.TOD.CKPT_PATH)

    #     if self.cfg.ACTIVE_LEARNING.AL_ITERATION == 1:
    #         # first time so random sampling
    #         return self.random(uSet, budgetSize)

    #     ## Load previous model
    #     prev_clf_path = self.cfg.TOD.CKPT_PATH
    #     import pycls.core.model_builder as model_builder
    #     prev_clf = model_builder.build_model(self.cfg, active_sampling=self.cfg.ACTIVE_LEARNING.ACTIVATE, isDistributed=False)

    #     checkpoint = torch.load(prev_clf_path, map_location='cpu')

    #     if 'model_state' in checkpoint:
    #         prev_clf.load_state_dict(checkpoint['model_state'])
    #     else:
    #         isModuleStrPresent=False
    #         #remove module
    #         for k in checkpoint.keys():
    #             if k.find("module.") == -1:
    #                 continue
    #             isModuleStrPresent=True
    #             break

    #         if isModuleStrPresent:
    #             print("Loaded checkpoint contains module present in keys.")
    #             print("So now removing 'module' strings")
    #             #remove module strings
    #             from collections import OrderedDict
    #             new_ckpt_dict = OrderedDict()
    #             for k,v in checkpoint.items():
    #                 tmp_key = k.replace("module.","")
    #                 new_ckpt_dict[tmp_key] = v

    #             checkpoint = copy.deepcopy(new_ckpt_dict)
    #             print("Done!!")

    #         prev_clf.load_state_dict(checkpoint)
    #     prev_clf.cuda(torch.cuda.current_device())

    #     prev_clf.eval()

    #     print('Loaded prev model weights from: {}'.format(prev_clf_path))

    #     u_ranks = []

    #     if self.cfg.TRAIN.DATASET == "IMAGENET":
    #         print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
    #         clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
    #         prev_clf = torch.nn.DataParallel(prev_clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
    #         uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
    #     else:
    #         uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE),data=dataset)

    #     n_uLoader = len(uSetLoader)
    #     print("len(uSetLoader): {}".format(n_uLoader))

    #     with torch.no_grad():
    #         for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
    #             x_u = x_u.cuda(0)

    #             temp_u_rank = clf(x_u)
    #             temp_u_rank_prev = prev_clf(x_u)

    #             #get probs
    #             temp_u_rank = torch.nn.functional.softmax(temp_u_rank, dim=1)
    #             temp_u_rank_prev = torch.nn.functional.softmax(temp_u_rank_prev, dim=1)

    #             # #find the max prob for having certainty of a clf
    #             # temp_u_rank, _ = torch.max(temp_u_rank, dim=1)

    #             pred_loss = (temp_u_rank - temp_u_rank_prev).pow(2).sum(1) / 2
    #             u_ranks.append(pred_loss.cpu().numpy())

    #     u_ranks = np.concatenate(u_ranks, axis=0)
    #     #Now u_ranks has shape: [U_Size x 1]

    #     #index of u_ranks serve as key to refer in u_idx
    #     print(f"u_ranks.shape: {u_ranks.shape}")

    #     # To manually inspect if sampling works correctly
    #     scores_save_path = self.cfg.OUT_DIR

    #     print('--------')
    #     print('score save path: ', scores_save_path)
    #     print('budgetsize: ', budgetSize)
    #     print('--------')
    #     os.makedirs(scores_save_path, exist_ok=True) # just to be safe
    #     with open(os.path.join(scores_save_path,"actualScores.txt"), 'w') as fpw:
    #         for temp_idx, temp_rank in zip(uSet, u_ranks):
    #             fpw.write(f'{temp_idx}\t{temp_rank:.6f}\n')

    #     fpw.close()

    #     #argsort helps to return the indices of u_ranks such that indices with large values comes at last
    #     sorted_idx = np.argsort(u_ranks)

    #     #reverse the sorted indexes such that indices with large values come in the beginning
    #     sorted_idx = sorted_idx[::-1]

    #     activeSet = sorted_idx[:budgetSize]

    #     activeSet = uSet[activeSet]
    #     remainSet = uSet[sorted_idx[budgetSize:]]

    #     # Update the prev ckpt tod path for next iteration

    #     # print('~~ Before TOD_CKPT_PATH: ', self.cfg.TOD.CKPT_PATH)
    #     # self.cfg.TOD.CKPT_PATH = self.cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR
    #     # print('~~ After TOD.CKPT_PATH: ', self.cfg.TOD.CKPT_PATH)

    #     temp_cfg = deepcopy(self.cfg)
    #     self.cfg.TOD.CKPT_PATH = self.cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR
    #     temp_cfg.TOD.CKPT_PATH = self.cfg.ACTIVE_LEARNING.MODEL_LOAD_DIR

    #     print('*** TOD.CKPT_PATH: ', self.cfg.TOD.CKPT_PATH)

    #     # Write cfg file
    #     self.dump_modified_cfg()

    #     return activeSet, remainSet
