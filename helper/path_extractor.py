import os
import numpy as np
import glob

# gets latest_model_path by sorting models wrt epoch
def get_latest_model_path(dir_path):
    """Returns the path of model which is saved with largest epoch

    Args:
        dir_path (str): Path to all the checkpoints.

    Returns:
        str: Path to the last (with respect to epochs) checkpoint.
    """
    all_model_paths = glob.glob("{}*.pyth".format(dir_path))
    print(all_model_paths)
    assert len(all_model_paths) != 0, "No models found at path {}".format(dir_path)
    model_names = [os.path.split(i)[1] for i in all_model_paths]
    # There are two splits as one splits get epoch in format: **[00xy].pyth
    # So the next split takes integer part form **.pyth
    epoch_nums = np.array([int(m.split("_")[-1].split(".")[0]) for m in model_names])
    max_epoch_idx = np.argmax(epoch_nums)
    return all_model_paths[max_epoch_idx]


def get_best_model_path(
    args, data_splits, i, directory_specific, alStart, directPath=""
):
    """Returns the path to best model based on the validation set performance.

    Args:
        args: Reference to input args.
        data_splits (list): Possible data splits in current experiment. For eg: with 10% for 4 AL iterations, data_splits = [10.0, 20.0, 30.0, 40.0]
        i (int): seed_id
        directory_specific (str): Specifies which strong regularization is applied (i.e swa_rand_aug or rand_aug) or nothing is applied at all (i.e vanilla)
        alStart (bool): Is it the first AL iteration or not.
        directPath (str, optional): if available, path to all the checkpoints to be considered. Defaults to "".

    Returns:
        str: Path to the model with best performance on  validation set.
    """
    # print('[at data_splits: ',data_splits[i],']directPath: ', directPath)
    if directPath == "":
        if alStart:

            print("~~~ args.out_dir: ", args.out_dir)

            # first time al so refer to random baseclf
            if "auto_ml_results" in os.listdir(args.out_dir):
                temp_path = "{}{}/{}/{}/{}/checkpoints/".format(
                    args.out_dir,
                    args.dataset,
                    args.init_partition,
                    args.model_type + "_depth_" + str(args.model_depth),
                    directory_specific,
                )
            else:
                temp_path = "{}{}/{}/{}/{}/{}/checkpoints/".format(
                    args.out_dir,
                    args.dataset,
                    args.init_partition,
                    str(args.seed_id),
                    args.model_type + "_depth_" + str(args.model_depth),
                    directory_specific,
                )

        else:
            # temp_path = "{}{}/{}/{}/{}/{}/{}/checkpoints/"\
            #     .format(args.out_dir, args.dataset,str(data_splits[i-1]),str(args.seed_id),args.model_type+'_depth_'+str(args.model_depth),directory_specific,args.sampling_fn)
            if args.init_partition != data_splits[i]:
                temp_path = "{}{}/{}/{}/{}/{}/{}".format(
                    args.out_dir,
                    args.dataset,
                    str(data_splits[i - 1]),
                    str(args.seed_id),
                    args.model_type + "_depth_" + str(args.model_depth),
                    directory_specific,
                    args.sampling_fn,
                )
            else:
                temp_path = "{}{}/{}/{}/{}/{}".format(
                    args.out_dir,
                    args.dataset,
                    str(data_splits[i - 1]),
                    str(args.seed_id),
                    args.model_type + "_depth_" + str(args.model_depth),
                    directory_specific,
                )
            temp_path = f"{temp_path}/checkpoints/"
    else:
        if directPath.find("checkpoints") == -1:
            temp_path = directPath + "checkpoints/"
        else:
            temp_path = directPath

    all_model_paths = glob.glob("{}*.pyth".format(temp_path))

    # because vaal and qbc can work without previous models present i.e the sampling process at ith data split
    # is independent of task model trained at i+1th split
    if args is not None and not (
        args.sampling_fn.startswith("ensemble") or args.sampling_fn.startswith("vaal")
    ):
        assert len(all_model_paths) != 0, "No models found at path {}".format(temp_path)

    if len(all_model_paths) == 0:
        return ""
    model_names = [os.path.split(i)[1] for i in all_model_paths]
    val_acc = np.array([m.split("_")[2] for m in model_names], dtype=float)
    best_val_acc_idx = np.argsort(val_acc)[-1]  # -1 to get index for max
    best_model_path = all_model_paths[best_val_acc_idx]

    print("best_model_path chosen: {}".format(best_model_path))
    return best_model_path


def search_best_model_path(temp_path: str, isDirectPath: bool = False):
    if isDirectPath:
        all_dirs = [temp_path]
    else:
        all_dirs = os.listdir(temp_path)
        all_dirs = [
            os.path.join(temp_path, temp_dir, "checkpoints") for temp_dir in all_dirs
        ]
        print(f"Number of trials found at {temp_path}: {len(all_dirs)}")

    all_model_paths = []
    for temp_dir in all_dirs:
        temp_model_paths = glob.glob(os.path.join(temp_dir, "*.pyth"))
        all_model_paths = all_model_paths + temp_model_paths

    model_names = [os.path.split(i)[1] for i in all_model_paths]
    val_acc = np.array([m.split("_")[2] for m in model_names], dtype=float)
    best_val_acc_idx = np.argsort(val_acc)[-1]  # -1 to get index for max
    best_model_path = all_model_paths[best_val_acc_idx]

    print("best_model_path chosen: {}".format(best_model_path))
    if isDirectPath:
        return val_acc[best_val_acc_idx]

    return best_model_path


def update_lset_uset_paths(al_args, lpath, upath):
    al_args[1] = lpath
    al_args[3] = upath
    return al_args
