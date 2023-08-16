import torchvision
import torch
from torchvision.datasets import (
    MNIST,
    CIFAR10,
    CIFAR100,
    SVHN,
    MNIST,
    STL10,
)  # ,ImageNet
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler  # , SequentialSampler
from al_utils.autoaugment import RandAugmentPolicy, SplitAugmentPolicy
from pycls.datasets.sampler import IndexedSequentialSampler, IndexedDistributedSampler


class Data:
    """
    Contains all data related functions. For working with new dataset
    make changes to following functions:
    1. getDataset
    2. getPreprocessOps
    3. getLoaders

    Note specify the dataset in CAPITAL Letters only.
    """

    def __init__(self, dataset, israndAug=False, args=None):
        """
        Initializes dataset attribute of (Data class) object with specified "dataset" argument.
        INPUT:
        dataset: String, Name of the dataset.
        """
        self.dataset = dataset
        self.eval_mode = False
        self.is_augmented = True  # cfg.SIMPLE_AUGMENTATIONS #Always True
        self.rand_augment = israndAug
        self.rand_augment_N = 1 if args is None else args.rand_aug_N
        self.rand_augment_M = 5 if args is None else args.rand_aug_M

        # cfg.RANDAUG.ACTIVATE
        # print("=========IN DATA CLASS INSTANTIATION==============")
        # print(f"self.rand_augment: {self.rand_augment}")

    def about(self):
        """
        Show all properties of this class.
        """
        print(self.__dict__)

    def getPreprocessOps(self):
        """
        This function specifies the steps to be accounted for preprocessing.

        INPUT:
        None

        OUTPUT:
        Returns a list of preprocessing steps. Note the order of operations matters in the list.
        """
        if self.dataset in [
            "MNIST",
            "CIFAR10",
            "CIFAR100",
            "IMAGENET",
            "SVHN",
            "MNIST",
            "STL10",
            "RSNA",
        ]:
            ops = []
            if self.is_augmented:
                print("Dataset is augmented")
                if self.dataset == "CIFAR100":
                    ops = [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                    ]
                elif self.dataset == "CIFAR10":
                    ops = [transforms.RandomHorizontalFlip()]
                elif self.dataset == "SVHN":
                    ops = [transforms.RandomCrop(32, padding=4)]
                elif self.dataset == "STL10":
                    # ops = [transforms.Resize(32),transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]
                    ops = [
                        transforms.RandomCrop(96, padding=12),
                        transforms.RandomHorizontalFlip(),
                    ]
                elif self.dataset == "MNIST":
                    ops = []

                elif self.dataset == "IMAGENET":
                    # ops = [transforms.Resize(256),transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]

                    # Source: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L205
                    ops = [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]
                    # import PIL
                    # ops = [transforms.Resize(256, interpolation=PIL.Image.BICUBIC), transforms.CenterCrop(224)]
                elif self.dataset == "RSNA":
                    ops = [
                        transforms.Resize(256, antialias=True),
                        transforms.RandomResizedCrop((224,224), antialias=True),
                    ]
                    
                else:
                    raise NotImplementedError

            # Rand_Augment
            if not self.eval_mode and self.rand_augment:
                # N and M values are taken from Experiment Section of RandAugment Paper
                # Though RandAugment paper works with WideResNet model
                if self.dataset == "CIFAR10":
                    # ops.append(RandAugmentPolicy(N=1,M=5))
                    ops.append(
                        RandAugmentPolicy(N=self.rand_augment_N, M=self.rand_augment_M)
                    )

                elif self.dataset == "CIFAR100":
                    # ops.append(RandAugmentPolicy(N=1, M=2))
                    ops.append(
                        RandAugmentPolicy(N=self.rand_augment_N, M=self.rand_augment_M)
                    )

                elif self.dataset == "STL10":
                    # ops.append(RandAugmentPolicy(N=1, M=2))
                    ops.append(
                        RandAugmentPolicy(N=self.rand_augment_N, M=self.rand_augment_M)
                    )

                elif self.dataset == "SVHN":
                    # ops.append(RandAugmentPolicy(N=1, M=5))
                    # Ref: N and M values specified in appendix of randuag paper: https://arxiv.org/pdf/1909.13719.pdf
                    ops.append(
                        RandAugmentPolicy(N=self.rand_augment_N, M=self.rand_augment_M)
                    )

                elif self.dataset == "MNIST":
                    ops.append(
                        RandAugmentPolicy(N=self.rand_augment_N, M=self.rand_augment_M)
                    )

                elif self.dataset == "IMAGENET":
                    # ops.append(RandAugmentPolicy(N=1, M=9))
                    ops.append(
                        RandAugmentPolicy(N=self.rand_augment_N, M=self.rand_augment_M)
                    )
                elif self.dataset == "RSNA":
                    ops.append(
                        RandAugmentPolicy(N=self.rand_augment_N, M=self.rand_augment_M)
                    )

            ops.append(transforms.ToTensor())
            if self.dataset == "RSNA":
                ops.append(transforms.Lambda(lambda x: x.expand(3,*x.shape[1:])),)

            if self.eval_mode:
                if self.dataset == "IMAGENET":
                    ops = [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                    ]
                elif self.dataset == "RSNA":
                    ops = [
                        transforms.Resize((224,224), antialias=True),
                        transforms.ToTensor(),             
                        transforms.Lambda(lambda x: x.expand(3,*x.shape[1:])),
                    ]
                else:
                    ops = [transforms.ToTensor()]

            print("--------------------------------------")
            print("Preprocess Operations Selected ==> ", ops)
            print("--------------------------------------")
            return ops
        else:
            print(
                "Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(
                    self.dataset
                )
            )
            raise NotImplementedError
            return

    def getDataset(self, save_dir, preprocess_steps=[], isTrain=True, isDownload=False):
        """
        This function returns the dataset instance and number of data points in it.

        INPUT:
        save_dir: String, It specifies the path where dataset will be saved if downloaded.

        preprocess_steps(optional): List, Contains the ordered operations used for preprocessing the data.

        isTrain (optional): Bool, If true then Train partition is downloaded else Test partition.

        isDownload (optional): Bool, If true then dataset is saved at path specified by "save_dir".

        OUTPUT:
        (On Success) Returns the tuple of dataset instance and length of dataset.
        (On Failure) Returns Message as <dataset> not specified.
        """

        if self.dataset == "MNIST":
            # if preprocess steps undefined
            if len(preprocess_steps) == 0:
                preprocess_steps = self.getPreprocessOps()

            preprocess_steps = transforms.Compose(preprocess_steps)
            mnist = MNIST(
                save_dir, train=isTrain, transform=preprocess_steps, download=isDownload
            )
            return mnist, len(mnist)

        elif self.dataset == "CIFAR10":
            # if preprocess steps undefined
            if len(preprocess_steps) == 0:
                preprocess_steps = self.getPreprocessOps()

            preprocess_steps = transforms.Compose(preprocess_steps)
            cifar10 = CIFAR10(
                save_dir, train=isTrain, transform=preprocess_steps, download=isDownload
            )
            return cifar10, len(cifar10)

        elif self.dataset == "CIFAR100":
            # if preprocess steps undefined
            if len(preprocess_steps) == 0:
                preprocess_steps = self.getPreprocessOps()

            preprocess_steps = transforms.Compose(preprocess_steps)
            cifar100 = CIFAR100(
                save_dir, train=isTrain, transform=preprocess_steps, download=isDownload
            )
            return cifar100, len(cifar100)

        elif self.dataset == "SVHN":
            # if preprocess steps undefined
            if len(preprocess_steps) == 0:
                preprocess_steps = self.getPreprocessOps()

            preprocess_steps = transforms.Compose(preprocess_steps)
            if isTrain:
                svhn = SVHN(
                    save_dir,
                    split="train",
                    transform=preprocess_steps,
                    download=isDownload,
                )
            else:
                svhn = SVHN(
                    save_dir,
                    split="test",
                    transform=preprocess_steps,
                    download=isDownload,
                )
            return svhn, len(svhn)

        elif self.dataset == "STL10":
            # if preprocess steps undefined
            if len(preprocess_steps) == 0:
                preprocess_steps = self.getPreprocessOps()

            preprocess_steps = transforms.Compose(preprocess_steps)
            if isTrain:
                stl10 = STL10(
                    save_dir,
                    split="train",
                    transform=preprocess_steps,
                    download=isDownload,
                )
            else:
                stl10 = STL10(
                    save_dir,
                    split="test",
                    transform=preprocess_steps,
                    download=isDownload,
                )
            return stl10, len(stl10)

        elif self.dataset == "IMAGENET":
            # if preprocess steps undefined
            if len(preprocess_steps) == 0:
                preprocess_steps = self.getPreprocessOps()

            preprocess_steps = transforms.Compose(preprocess_steps)
            imagenet = torchvision.datasets.ImageFolder(
                root=save_dir, transform=preprocess_steps
            )
            # imagenet = ImageNet(save_dir,train=isTrain,transform=preprocess_steps,download=isDownload)
            return imagenet, len(imagenet)
        
        elif self.dataset == "RSNA":
            # if preprocess steps undefined
            if len(preprocess_steps) == 0:
                preprocess_steps = self.getPreprocessOps()
            
            # if isDownload:
            #     print("Warning: Download RSNA Dataset is not supported. Please run `download_data.sh`. Ignore this if you already download")
                # raise NotImplementedError

            from pycls.datasets.rsna import RSNA

            rsna = RSNA(
                data_path=save_dir, 
                csv_path="data/RSNA/stage_2_train_labels.csv",
                transforms=transforms.Compose(preprocess_steps)
            )
            return rsna, len(rsna)

        else:
            print(
                "Either the specified {} dataset is not added or there is no if condition in getDataset function of Data class".format(
                    self.dataset
                )
            )
            raise NotImplementedError

    def getDatasetForVAAL(
        self, save_dir, preprocess_steps=[], isTrain=True, isDownload=False
    ):
        # if preprocess steps undefined
        if len(preprocess_steps) == 0:
            preprocess_steps = ops = [transforms.Resize(32), transforms.ToTensor()]

        preprocess_steps = transforms.Compose(preprocess_steps)
        if isTrain:
            stl10 = STL10(
                save_dir, split="train", transform=preprocess_steps, download=isDownload
            )
        else:
            stl10 = STL10(
                save_dir, split="test", transform=preprocess_steps, download=isDownload
            )
        return stl10, len(stl10)

    def getLUIndexesList(self, train_split_ratio, val_split_ratio, data, seed_id):
        """
        Initialize the labelled and unlabelled set by splitting the data into train
        and validation according to split_ratios arguments.

        Visually it does the following:

        |<------------- Train -------------><--- Validation --->

        |<--- Labelled --><---Unlabelled --><--- Validation --->

        INPUT:
        train_split_ratio: Float, Specifies the proportion of data in train set.
        For example: 0.8 means beginning 80% of data is training data.

        val_split_ratio: Float, Specifies the proportion of data in validation set.
        For example: 0.1 means ending 10% of data is validation data.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT:
        (On Success) Sets the labelled, unlabelled set along with validation set
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        assert isinstance(
            train_split_ratio, float
        ), "Train split ratio is of {} datatype instead of float".format(
            type(train_split_ratio)
        )
        assert isinstance(
            val_split_ratio, float
        ), "Val split ratio is of {} datatype instead of float".format(
            type(val_split_ratio)
        )
        assert self.dataset in [
            "MNIST",
            "CIFAR10",
            "CIFAR100",
            "IMAGENET",
            "SVHN",
            "STL10",
            "RSNA",
        ], "Sorry the dataset {} is not supported. Currently we support ['MNIST','CIFAR10']".format(
            self.dataset
        )

        lSet = []
        uSet = []
        valSet = []

        n_dataPoints = len(data)
        all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(all_idx)
        train_splitIdx = int(train_split_ratio * n_dataPoints)
        # To get the validation index from end we multiply n_datapoints with 1-val_ratio
        val_splitIdx = int((1 - val_split_ratio) * n_dataPoints)
        # Check there should be no overlap with train and val data
        assert (
            train_split_ratio + val_split_ratio < 1.0
        ), "Validation data over laps with train data as last train index is {} and last val index is {}. \
            The program expects val index > train index. Please satisfy the constraint: train_split_ratio + val_split_ratio < 1.0; currently it is {} + {} is not < 1.0 => {} is not < 1.0".format(
            train_splitIdx,
            val_splitIdx,
            train_split_ratio,
            val_split_ratio,
            train_split_ratio + val_split_ratio,
        )

        lSet = all_idx[:train_splitIdx]
        uSet = all_idx[train_splitIdx:val_splitIdx]
        valSet = all_idx[val_splitIdx:]

        print(
            "lSet len: {}, uSet len: {} and valSet len: {}".format(
                len(lSet), len(uSet), len(valSet)
            )
        )
        lSet = np.array(lSet, dtype=np.ndarray)
        uSet = np.array(uSet, dtype=np.ndarray)
        valSet = np.array(valSet, dtype=np.ndarray)
        return lSet, uSet, valSet

    def getSequentialDataLoader(self, indexes, batch_size, data):
        """
        Gets reference to the data loader which provides batches of <batch_size> sequentially
        from indexes set. We use SubsetRandomSampler as sampler in returned DataLoader.

        ARGS
        -----

        indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

        batch_size: int, Specifies the batchsize used by data loader.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT
        ------

        Returns a reference to dataloader
        """

        assert isinstance(
            indexes, np.ndarray
        ), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indexes))
        assert isinstance(
            batch_size, int
        ), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(
            type(batch_size)
        )

        subsetSampler = IndexedSequentialSampler(indexes)

        if self.dataset == "IMAGENET":
            loader = DataLoader(
                dataset=data,
                batch_size=batch_size,
                sampler=subsetSampler,
                pin_memory=True,
            )
        else:
            loader = DataLoader(
                dataset=data, batch_size=batch_size, sampler=subsetSampler
            )
        return loader

    def getIndexesDataLoader(self, indexes, batch_size, data, num_workers=0):
        """
        Gets reference to the data loader which provides batches of <batch_size> by randomly sampling
        from indexes set. We use SubsetRandomSampler as sampler in returned DataLoader.

        ARGS
        -----

        indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

        batch_size: int, Specifies the batchsize used by data loader.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT
        ------

        Returns a reference to dataloader
        """

        assert isinstance(
            indexes, np.ndarray
        ), "Indexes has dtype: {} whereas expected is nd.array.".format(type(indexes))
        assert isinstance(
            batch_size, int
        ), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(
            type(batch_size)
        )

        subsetSampler = SubsetRandomSampler(indexes)
        # print(data)
        if self.dataset == "IMAGENET":
            loader = DataLoader(
                dataset=data,
                batch_size=batch_size,
                sampler=subsetSampler,
                pin_memory=False,
                num_workers=num_workers,
            )
        else:
            loader = DataLoader(
                dataset=data,
                batch_size=batch_size,
                sampler=subsetSampler,
                num_workers=num_workers,
            )
        return loader

    def getDistributedIndexesDataLoader(
        self,
        cfg,
        indexes,
        batch_size,
        data,
        n_worker=4,
        pin_memory=True,
        drop_last=True,
        allowRepeat=True,
        shuffle=True,
    ):
        """
        Gets reference to the distribured data loader which provides batches of <batch_size> by randomly sampling
        from indexes set. We use SubsetRandomSampler as sampler in returned DataLoader.
        Best Performance when used in conjunction to torch.nn.DistributedDataParallel

        ARGS
        -----

        indexes: np.ndarray, dtype: int, Array of indexes which will be used for random sampling.

        batch_size: int, Specifies the batchsize used by data loader.

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        OUTPUT
        ------

        Returns a reference to dataloader
        """

        assert isinstance(
            batch_size, int
        ), "Batchsize is expected to be of int type whereas currently it has dtype: {}".format(
            type(batch_size)
        )

        if indexes is None:
            # used for loading test dataset in CIFAR10/100
            temp_indexes = [i for i in range(len(data))]
            subsetSampler = SubsetRandomSampler(temp_indexes)
        else:
            temp_indexes = indexes

        temp_indexes = np.array(temp_indexes)

        if not cfg.NUM_GPUS > 1:  # or (allowRepeat == False and indexes is None):
            return self.getIndexesDataLoader(
                indexes=temp_indexes, batch_size=batch_size, data=data
            )
        else:
            subsetSampler = IndexedDistributedSampler(
                dataset=data, index_set=temp_indexes, allowRepeat=allowRepeat
            )  # if cfg.NUM_GPUS > 1 else None
        print("=============================")
        print("IndexedDistributedSampler data sampler")
        print(
            "sampler: {} where repeatAllow: {} and dropLast: {}".format(
                subsetSampler, allowRepeat, drop_last
            )
        )
        print("=============================")

        # Create a loader
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=(False if subsetSampler else shuffle),
            sampler=subsetSampler,
            num_workers=n_worker,
            pin_memory=True if self.dataset == "IMAGENET" else pin_memory,
            drop_last=drop_last,
        )

        # print(data)
        # if self.dataset == "IMAGENET":
        #     loader = DataLoader(dataset=data, batch_size=batch_size,sampler=subsetSampler, pin_memory=True)
        # else:
        #     loader = DataLoader(dataset=data, batch_size=batch_size,sampler=subsetSampler)
        return loader

    def getLoaders(self, split_ratio, data, batch_size, val_ratio, seed_id):

        """
        Splits the data into train and validation according to "split_ratio" argument.

        INPUT:
        split_ratio: Float, Specifies the proportion of data in train set.
        For example: 0.8 means 80% of data is training data and rest is for validation data

        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        batch_size: np.array, 1D, Contains two elements where first is batch size for train data loader and other denotes batch size for validation data loader.

        seed_id: int, Helps in reporoducing results of random operations

        val_ratio: float, Index corresponding to last portion
        For example, val_ratio is 0.1 then index is beggining index of last 10% of data.

        OUTPUT:
        (On Success) Returns the trainLoader, valLoader
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        if self.dataset in ["MNIST", "CIFAR10", "CIFAR100", "SVHN", "STL10", "RSNA"]:
            n_datapts = len(data)
            idx = [i for i in range(n_datapts)]
            splitIdx = int(split_ratio * n_datapts)
            clf_val_splitIdx = int((1 - opts["clf_val_split_ratio"]) * n_datapts)
            np.random.shuffle(idx)

            train_idx = idx[:splitIdx]
            val_idx = idx[clf_val_splitIdx:]

            tr_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_batch_size, val_batch_size = int(batch_size[0]), int(batch_size[1])
            trainLoader = DataLoader(
                data, batch_size=train_batch_size, sampler=tr_sampler
            )
            valLoader = DataLoader(data, batch_size=val_batch_size, sampler=val_sampler)

            trainDataPoints = len(train_idx)
            valDataPoints = len(val_idx)
            print(
                "Train Data Splitted. Train datapoints: [{}/{}], Validation datapoints: [{}/{}]".format(
                    trainDataPoints, n_datapts, valDataPoints, n_datapts
                )
            )

            return trainLoader, valLoader

        else:
            print(
                "Either the specified {} dataset is not added or there is no if condition in getLoaders function of Data class".format(
                    self.dataset
                )
            )
            raise NotImplementedError
            return

    def getInfiniteBatches(self, dataloader, labels=False):
        if labels:
            while True:
                for _, (img, label) in enumerate(dataloader):
                    yield img, label
        else:
            while True:
                for _, (img, _) in enumerate(dataloader):
                    yield img

    def getTestLoader(self, data, test_batch_size, seed_id=0):
        """
        Implements a random subset sampler for sampling the data from test set.

        INPUT:
        data: reference to dataset instance. This can be obtained by calling getDataset function of Data class.

        test_batch_size: int, Denotes the size of test batch

        seed_id: int, Helps in reporoducing results of random operations

        OUTPUT:
        (On Success) Returns the testLoader
        (On Failure) Returns Message as <dataset> not specified.
        """
        # Reproducibility stuff
        torch.manual_seed(seed_id)
        np.random.seed(seed_id)

        if self.dataset in ["MNIST", "CIFAR10", "CIFAR100"]:
            n_datapts = len(data)
            idx = [i for i in range(n_datapts)]
            # np.random.shuffle(idx)

            test_sampler = SubsetRandomSampler(idx)

            testLoader = DataLoader(
                data, batch_size=test_batch_size, sampler=test_sampler
            )
            return testLoader

        else:
            raise NotImplementedError

    def loadPartitions(self, lSetPath, uSetPath, valSetPath):

        assert isinstance(lSetPath, str), "Expected lSetPath to be a string."
        assert isinstance(uSetPath, str), "Expected uSetPath to be a string."
        assert isinstance(valSetPath, str), "Expected lSetPath to be a string."

        lSet = np.load(lSetPath, allow_pickle=True)
        uSet = np.load(uSetPath, allow_pickle=True)
        valSet = np.load(valSetPath, allow_pickle=True)

        # Checking no overlap
        assert (
            len(set(valSet) & set(uSet)) == 0
        ), "Intersection is not allowed between validationset and uset"
        assert (
            len(set(valSet) & set(lSet)) == 0
        ), "Intersection is not allowed between validationset and lSet"
        assert (
            len(set(uSet) & set(lSet)) == 0
        ), "Intersection is not allowed between uSet and lSet"

        return lSet, uSet, valSet

    def getClassWeightsFromDataset(self, dataset, index_set, bs):
        temp_loader = self.getIndexesDataLoader(
            indexes=index_set, batch_size=bs, data=dataset
        )
        return self.getClassWeights(temp_loader)

    def getClassWeights(self, dataloader):

        """
        INPUT
        dataloader: dataLoader

        OUTPUT
        Returns a tensor of size C where each element at index i represents the weight for class i.
        """

        all_labels = []
        for _, y in dataloader:
            all_labels.append(y)
        print("===Computing Imbalanced Weights===")

        all_labels = np.concatenate(all_labels, axis=0)
        print(f"all_labels.shape: {all_labels.shape}")
        classes = np.unique(all_labels)
        print(f"classes: {classes.shape}")
        num_classes = len(classes)
        freq_count = np.zeros(num_classes, dtype=int)
        for i in classes:
            freq_count[i] = (all_labels == i).sum()

        # Normalize
        freq_count = (1.0 * freq_count) / np.sum(freq_count)
        print(f"=== Sum(freq_count): {np.sum(freq_count)} ===")
        class_weights = 1.0 / freq_count

        class_weights = torch.Tensor(class_weights)
        return class_weights
