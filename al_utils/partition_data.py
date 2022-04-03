from data import Data
import argparse
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from power_law import y_function


def main(args, imbalanced=False):
    print(args)
    dataObj = Data(dataset=args.dataset, args=args)
    if args.dataset == "IMAGENET":
        # Imagenet is loaded from args.data_dir_path
        trainDataset, n_TrainDatapts = dataObj.getDataset(
            save_dir=args.data_dir, isTrain=True, isDownload=True
        )
    else:
        trainDataset, n_TrainDatapts = dataObj.getDataset(
            save_dir="./data/" + args.dataset + "/" + args.data_dir,
            isTrain=True,
            isDownload=True,
        )
    # print(trainDataset)
    print("Number of total datapoints in train Set: {}".format(n_TrainDatapts))

    if imbalanced:
        all_labels = trainDataset.targets
        all_labels = np.array(all_labels)
        print("len(all_labels): {}".format(len(all_labels)))
        classes = list(trainDataset.class_to_idx.values())
        print(classes)
        freq_map = np.arange(len(classes))

        if args.dataset == "CIFAR100":
            temp_k = 400
            temp_alpha = -0.046
            temp_a = 100
        elif args.dataset == "CIFAR10":
            temp_k = 4000
            temp_alpha = -0.32
            temp_a = 500
        else:
            raise NotImplementedError

        j = 0.5  # indexing on x-axis for exponential decay function
        for i in range(len(classes)):
            num_samples_i = y_function(k=temp_k, x=j, alpha=temp_alpha, a=temp_a)
            freq_map[i] = num_samples_i
            j += 1

        min_freq_map = np.min(freq_map)

        plt.bar(x=np.arange(len(classes)), height=freq_map)
        plt.title(
            "k={}, alpha={}, a={}; Min Freq: {}".format(
                temp_k, temp_alpha, temp_a, min_freq_map
            )
        )

        plot_save_dir = (
            args.save_dir
            + "final_distribution_k_{}_alpha_{}_a_{}".format(temp_k, temp_alpha, temp_a)
            + ".png"
        )
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        plt.savefig(plot_save_dir)
        print("Saved distribution plot at path: {}".format(plot_save_dir))

        freq_map = freq_map / np.sum(freq_map)  # Normalize in range [0..1]

        ##Partition Data
        lSet_len = 5000
        freq_map = freq_map * lSet_len
        freq_map = np.array(freq_map + 0.5, dtype=int)

        lSet = []
        n_classes = len(trainDataset.classes)
        print("Number of classes: {}".format(n_classes))
        for i in range(n_classes):
            label_idx = np.where(all_labels == i)[0]

            choice_label_idx = list(
                np.random.choice(label_idx, freq_map[i], replace=False)
            )
            # print(len(label_idx)," ~sampled-> ",freq_map[i], "actually sampled: ",len(choice_label_idx)," idx: ",choice_label_idx)
            assert (
                len(set(lSet) & set(choice_label_idx)) == 0
            ), "overlap in lSet and newly chosen set"
            lSet += choice_label_idx

        remainSet = np.arange(n_TrainDatapts)
        remainSet = set(remainSet) - set(lSet)  # set difference
        remainSet = list(remainSet)

        print("len(lSet): {}".format(len(lSet)))
        print("len(remainSet): {}".format(len(remainSet)))

        # Shuffle the remaining set
        np.random.seed(seed=args.seed_id)
        np.random.shuffle(remainSet)

        uSet = remainSet[0:40000]
        valSet = remainSet[40000:]
        print(
            "[Imbalanced] Lset: {}, uSet: {}, valSet: {}".format(
                len(lSet), len(uSet), len(valSet)
            )
        )

    else:
        # Partition train data into lSet, uSet and valSet
        lSet, uSet, valSet = dataObj.getLUIndexesList(
            train_split_ratio=args.train_ratio,
            val_split_ratio=args.val_ratio,
            data=trainDataset,
            seed_id=args.seed_id,
        )

    assert (
        len(set(valSet) & set(uSet)) == 0
    ), "Intersection is not allowed between validationset and uset"
    assert (
        len(set(valSet) & set(lSet)) == 0
    ), "Intersection is not allowed between validationset and lSet"
    assert (
        len(set(uSet) & set(lSet)) == 0
    ), "Intersection is not allowed between uSet and lSet"
    # Save indices set for future use
    nameIdxSet = ["lSet", "uSet", "valSet"]
    idxSet = [lSet, uSet, valSet]
    dir_path = ""
    for i, temp_idx in enumerate(idxSet):
        if not args.save_dir == "":
            dir_path = args.save_dir
        else:
            dir_path = (
                "data/" + args.dataset + "/partition_" + str(args.partition_num) + "/"
            )
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(dir_path + nameIdxSet[i] + "_" + args.dataset + ".npy", temp_idx)
        print(
            "Saving {} index set at {}".format(
                nameIdxSet[i], dir_path + nameIdxSet[i] + "_" + args.dataset + ".npy"
            )
        )
    print("Saved dataset partition {}".format(args.partition_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Specify dataset for training. Options [CIFAR10|CIFAR100]",
    )
    parser.add_argument(
        "--partition_num", type=int, required=True, default=1, help="Partition number"
    )
    parser.add_argument(
        "--data_dir", type=str, default="train-CIFAR10", help="Path to save dataset"
    )
    parser.add_argument(
        "--seed_id",
        type=int,
        default=0,
        help="Seed value for reproducing the results.\
        Kindly maintain consistency with this among different experiments",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.1,
        help="Initial proportion of data to be considered as training data.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Last proportion of data to be considered as valudation data.",
    )
    parser.add_argument(
        "--augmented", action="store_true", help="If dataset needs to be augmented"
    )
    parser.add_argument(
        "--rand_augment", action="store_true", help="If random augmentation is applied"
    )
    parser.add_argument(
        "--save_dir", type=str, default="", help="Path to save indexes file"
    )
    parser.add_argument(
        "--rand_aug_N",
        type=int,
        default=1,
        help="Number of randaug transfromations applied to single input",
    )
    parser.add_argument(
        "--rand_aug_M",
        type=int,
        default=5,
        help="Magnitude for augmentation operations",
    )
    args = parser.parse_args()

    if not os.path.exists(".data/" + args.dataset + "/" + args.data_dir):
        os.makedirs(".data/" + args.dataset + "/" + args.data_dir)

    main(args, imbalanced=False)
