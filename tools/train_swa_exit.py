from al_utils.swa_util import swa_train

import sys
import pickle
import os

tempArgsFile = sys.argv[1]
# print("tempArgsFile received: {}".format(tempArgsFile))
# print("tempArgsFile exists check: {}".format(os.path.isfile(tempArgsFile)))

# Getting back the objects:
with open(tempArgsFile, "rb") as f:  # Python 3: open(..., 'rb')
    (
        args,
        latest_model_path,
        temp_l_SetPath,
        temp_u_SetPath,
        temp_out_dir,
        trainDataset,
        noAugDataset,
        cfg,
    ) = pickle.load(f)
    print(f"In train_swa_exit function: cfg.TRAIN.DATASET: {cfg.TRAIN.DATASET}")

swa_train(
    args,
    latest_model_path,
    temp_l_SetPath,
    temp_u_SetPath,
    temp_out_dir,
    trainDataset,
    noAugDataset,
    cfg,
)
