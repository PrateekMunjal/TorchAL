import torch
import os
import math
import numpy as np
from copy import deepcopy
from pycls.core.config import cfg
import pycls.utils.distributed as du
from tqdm import tqdm


class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget
        self.cuda_id = torch.cuda.current_device()

    def compute_dists(self, X, X_train):
        dists = (
            -2 * np.dot(X, X_train.T)
            + np.sum(X_train**2, axis=1)
            + np.sum(X**2, axis=1)[:, np.newaxis]
        )
        return dists

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(
            self.compute_dists(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled),
            axis=0,
        )
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        temp_range = 1000
        for j in range(1, labeled.shape[0], temp_range):
            if j + temp_range < labeled.shape[0]:
                dist = self.compute_dists(labeled[j : j + temp_range, :], unlabeled)
            else:
                # for last iteration only :)
                dist = self.compute_dists(labeled[j:, :], unlabeled)
                # dist = pairwise_distances(labeled[j:, :], unlabeled,metric='euclidean')
            min_dist = np.vstack(
                (min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1])))
            )
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

        amount = cfg.ACTIVE_LEARNING.BUDGET_SIZE - 1
        for i in range(amount):
            if i is not 0 and i % 500 == 0:
                print("{} Sampled out of {}".format(i, amount + 1))
            # dist = pairwise_distances(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled, metric='euclidean')
            dist = self.compute_dists(
                unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])),
                unlabeled,
            )
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        return greedy_indices, remainSet

    def get_vae_activations(self, vae, dataLoader):
        acts = []
        vae.eval()

        temp_max_iter = len(dataLoader)
        print("len(dataloader): {}".format(temp_max_iter))
        temp_iter = 0
        for x, y in dataLoader:
            x = x.type(torch.cuda.FloatTensor)
            x = x.cuda(self.cuda_id)
            _, _, mu, _ = vae(x)
            acts.append(mu.cpu().numpy())
            if temp_iter % 100 == 0:
                print(f"Iteration [{temp_iter}/{temp_max_iter}] Done!!")

            temp_iter += 1

        acts = np.concatenate(acts, axis=0)
        return acts

    def get_predictions(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []

        assert vae.training == False, "Expected vae model to be in eval mode"
        assert (
            discriminator.training == False
        ), "Expected discriminator model to be in eval mode"

        temp_idx = 0
        for images, _ in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]

        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        all_preds = all_preds.cpu().numpy()
        return all_preds

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

    def efficient_compute_dists(self, labeled, unlabeled):
        """ """
        N_L = labeled.shape[0]
        N_U = unlabeled.shape[0]
        dist_matrix = None

        temp_range = 1000

        unlabeled = torch.from_numpy(unlabeled).cuda(self.cuda_id)
        temp_dist_matrix = np.empty((N_U, temp_range))
        # for i in range(0, N_L, temp_range):
        for i in tqdm(range(0, N_L, temp_range), desc="Computing Distance Matrix"):
            end_index = i + temp_range if i + temp_range < N_L else N_L
            temp_labeled = labeled[i:end_index, :]
            temp_labeled = torch.from_numpy(temp_labeled).cuda(self.cuda_id)
            temp_dist_matrix = self.gpu_compute_dists(unlabeled, temp_labeled)
            temp_dist_matrix = torch.min(temp_dist_matrix, dim=1)[0]
            temp_dist_matrix = torch.reshape(
                temp_dist_matrix, (temp_dist_matrix.shape[0], 1)
            )
            if dist_matrix is None:
                dist_matrix = temp_dist_matrix
            else:
                dist_matrix = torch.cat((dist_matrix, temp_dist_matrix), dim=1)
                dist_matrix = torch.min(dist_matrix, dim=1)[0]
                dist_matrix = torch.reshape(dist_matrix, (dist_matrix.shape[0], 1))

        return dist_matrix.cpu().numpy()

    @torch.no_grad()
    def vae_sample_for_labeling(
        self, vae, uSet, lSet, unlabeled_dataloader, lSetLoader
    ):

        vae.eval()
        print("Computing activattions for uset....")
        u_scores = self.get_vae_activations(vae, unlabeled_dataloader)
        print("Computing activattions for lset....")
        l_scores = self.get_vae_activations(vae, lSetLoader)

        print("l_scores.shape: ", l_scores.shape)
        print("u_scores.shape: ", u_scores.shape)

        # dist_matrix = self.compute_dists(u_scores, l_scores)
        dist_matrix = self.efficient_compute_dists(l_scores, u_scores)
        print("Dist_matrix.shape: ", dist_matrix.shape)

        min_scores = np.min(dist_matrix, axis=1)
        sorted_idx = np.argsort(min_scores)[::-1]

        activeSet = uSet[sorted_idx[0 : self.budget]]
        remainSet = uSet[sorted_idx[self.budget :]]

        return activeSet, remainSet

    def sample_vaal_plus(self, vae, disc_task, data, cuda):
        all_preds = []
        all_indices = []

        assert vae.training == False, "Expected vae model to be in eval mode"
        assert (
            disc_task.training == False
        ), "Expected disc_task model to be in eval mode"

        temp_idx = 0
        for images, _ in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds, _ = disc_task(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]

        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(
            all_indices
        ), " Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet, uSet

    def sample(self, vae, discriminator, data, uSet, cfg):
        all_preds = []
        all_indices = []

        assert vae.training == False, "Expected vae model to be in eval mode"
        assert (
            discriminator.training == False
        ), "Expected discriminator model to be in eval mode"

        temp_idx = 0
        for images, _ in tqdm(data, desc="Constructing VAE ActiveSet"):
            images = images.type(torch.cuda.FloatTensor)
            images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]

        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)

        scores_save_path = cfg.OUT_DIR
        os.makedirs(scores_save_path, exist_ok=True)  # just to be safe
        with open(os.path.join(scores_save_path, "actualScores.txt"), "w") as fpw:
            for temp_idx, temp_rank in zip(uSet, all_preds):
                fpw.write(f"{temp_idx}\t{temp_rank:.6f}\n")

        fpw.close()

        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(
            all_indices
        ), " Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet, uSet

    # def sample_for_labeling(self, cfg, uSetPath, lSetPath, dataObj, noAugDataset):
    #     """
    #     Picks samples from uSet to form activeSet.

    #     INPUT
    #     ------
    #     vae: object of model VAE

    #     discriminator: object of model discriminator

    #     unlabeled_dataloader: Sequential dataloader iterating over uSet

    #     uSet: Collection of unlabelled datapoints

    #     NOTE: Please pass the unlabelled dataloader as sequential dataloader else the
    #     results won't be appropriate.

    #     OUTPUT
    #     -------

    #     Returns activeSet, [remaining]uSet
    #     """

    #     current_device = torch.cuda.current_device()

    #     #Load vae -- out_dir/vae.pyth
    #     vae_dir = os.path.join(cfg.OUT_DIR, "vae/vae.pyth")

    #     #Load disc -- out_dir/disc.pyth
    #     disc_dir = os.path.join(cfg.OUT_DIR, "disc/disc.pyth")

    #     #Get uSet form uSetPath
    #     uSet = np.load(uSetPath, allow_pickle=True)

    #     #Get uSetLoader from uSet
    #     uSetLoader = dataObj.getSequentialDataLoader(indexes=uSet,batch_size=int(cfg.TRAIN.BATCH_SIZE/cfg.NUM_GPUS),\
    #         data=noAugDataset)

    #     #load vae from vae_dir
    #     vae_checkpoint = None#load from vae_dir
    #     vae = torch.load(vae_checkpoint['model'], map_location='cpu')
    #     vae.cuda(current_device)

    #     #load disc from disc_dir
    #     disc_checkpoint = None
    #     disc = torch.load(disc_checkpoint['model'], map_location='cpu')
    #     disc.cuda(current_device)

    #     sampler = AdversarySampler(cfg.ACTIVE_LEARNING.BUDGET_SIZE)
    #     activeSet, remainSet = sampler.sample(vae, disc, uSetLoader)

    #     activeSet = uSet[activeSet]
    #     remainSet = uSet[remainSet]
    #     return activeSet, remainSet

    @torch.no_grad()
    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, uSet, cfg):
        """
        Picks samples from uSet to form activeSet.

        INPUT
        ------
        vae: object of model VAE

        discriminator: object of model discriminator

        unlabeled_dataloader: Sequential dataloader iterating over uSet

        uSet: Collection of unlabelled datapoints

        NOTE: Please pass the unlabelled dataloader as sequential dataloader else the
        results won't be appropriate.

        OUTPUT
        -------

        Returns activeSet, [remaining]uSet
        """
        print("Sampling....")
        activeSet, remainSet = self.sample(
            vae,
            discriminator,
            unlabeled_dataloader,
            uSet,
            cfg,
        )

        activeSet = uSet[activeSet]
        remainSet = uSet[remainSet]
        return activeSet, remainSet

    # def vaal_sampling(self, cfg, uSetPath, lSetPath, dataObj, noAugDataset):

    #     lSet = np.load(lSetPath, allow_pickle=True)
    #     uSet = np.load(uSetPath, allow_pickle=True)

    #     activeSet, remainSet = self.sample_for_labeling(cfg, uSetPath, lSetPath, dataObj, noAugDataset)

    #     lSet = np.append(lSet, activeSet)
    #     uSet = remainSet

    #     #save all sets
    #     np.save(os.path.join(cfg.OUT_DIR, "lSet.npy"), lSet)
    #     np.save(os.path.join(cfg.OUT_DIR, "uSet.npy"), uSet)
    #     np.save(os.path.join(cfg.OUT_DIR, "activeSet.npy"), activeSet)
