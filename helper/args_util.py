import argparse
import sys

from helper.path_extractor import get_best_model_path


def get_path_directory(alStart, args, data_split, directory_specific, i=0):
    """Generates basic path template for saving the results."""

    if alStart:
        if i == 1:
            return (
                args.out_dir
                + args.dataset
                + "/"
                + str(data_split)
                + "/"
                + str(args.seed_id)
                + "/"
                + args.sampling_fn
            )
        else:
            return "{}{}/{}/{}/{}/{}/{}/".format(
                args.out_dir,
                args.dataset,
                str(data_split),
                str(args.seed_id),
                args.model_type + "_depth_" + str(args.model_depth),
                directory_specific,
                args.sampling_fn,
            )
    else:
        return "{}{}/{}/{}/{}/{}/{}/".format(
            args.out_dir,
            args.dataset,
            str(data_split),
            str(args.seed_id),
            args.model_type + "_depth_" + str(args.model_depth),
            directory_specific,
            args.sampling_fn,
        )


def get_main_args(args):
    """Returns the list of main arguments which are used for training both baseclf model and active learning"""

    main_args = [
        "MODEL.NUM_CLASSES",
        args.num_classes,
        "MODEL.TYPE",
        args.model_type,
        "MODEL.DEPTH",
        args.model_depth,
        "NUM_GPUS",
        args.n_GPU,
        "OPTIM.BASE_LR",
        args.lr,
        "OPTIM.WEIGHT_DECAY",
        args.wt_decay,
        "OPTIM.MAX_EPOCH",
        args.clf_epochs,
        "OPTIM.MOMENTUM",
        args.optim_mom,
        "OPTIM.NESTEROV",
        args.nesterov,
        "PORT",
        args.port,
        "RNG_SEED",
        args.seed_id,
        "TEST.BATCH_SIZE",
        args.test_batch_size,
        "TEST.DATASET",
        args.dataset,
        "TRAIN.BATCH_SIZE",
        args.train_batch_size,
        "TRAIN.DATASET",
        args.dataset,
        "TRAIN.EVAL_PERIOD",
        args.eval_period,
        "TRAIN.CHECKPOINT_PERIOD",
        args.checkpoint_period,
        "TRAIN.IMBALANCED",
        args.isimbalanced,  # Transfer Experiment Hyperparams
        "TRAIN.TRANSFER_EXP",
        args.isTransferExp,
        "MODEL.TRANSFER_MODEL_TYPE",
        args.transfer_model_type,
        "MODEL.TRANSFER_MODEL_DEPTH",
        args.transfer_model_depth,
        "MODEL.TRANSFER_MODEL_STYLE",
        args.transfer_model_style,
        "MODEL.TRANSFER_DIR_SPECIFIC",
        args.transfer_dir_specific,  #'SIMPLE_AUGMENTATIONS', args.simple_augmentations, \
        "TRAIN_DIR",
        args.train_dir,
        "TEST_DIR",
        args.test_dir,
        "SWA_MODE.ACTIVATE",
        args.swa_mode,
        "SWA_MODE.LR",
        args.swa_lr,
        "SWA_MODE.START_ITER",
        args.swa_iter,
        "SWA_MODE.FREQ",
        args.swa_freq,
        "RANDAUG.ACTIVATE",
        args.rand_aug,
        "RANDAUG.N",
        args.rand_aug_N,
        "RANDAUG.M",
        args.rand_aug_M,
        "OPTIM.TYPE",
        args.optim,
        "LOG_PERIOD",
        args.log_iter,
        "VAAL.TRAIN_VAAL",
        args.sampling_fn in ["vaal", "vaal_minus_disc"],
        "VAAL.VAE_EPOCHS",
        args.vaal_epochs,
        "VAAL.VAE_LR",
        args.vaal_vae_lr,
        "VAAL.DISC_LR",
        args.vaal_disc_lr,
        "VAAL.BETA",
        args.vaal_beta,
        "VAAL.ADVERSARY_PARAM",
        args.vaal_adv_param,
        "VAAL.VAE_BS",
        args.vaal_vae_bs,
        "VAAL.Z_DIM",
        args.vaal_z_dim,
        "VAAL.IM_SIZE",
        args.vaal_im_size,
        "ENSEMBLE.MAX_EPOCH",
        args.ens_epochs,
        "DATA_LOADER.NUM_WORKERS",
        args.n_workers,
    ]
    return main_args


# all command line arguments defined here
def parse_args():

    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a classification model")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    # parser.add_argument('opts',help='See pycls/core/config.py for all options',default=None, nargs=argparse.REMAINDER)

    # We use train batch size as val batch size
    parser.add_argument(
        "--n_GPU", type=int, default=1, help="Number of Gpus used for parallelism"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for mulitprocess group creation. Note this should be unique for two process groups",
        default=10002,
    )
    # Always_end_path of out_dir with '/'
    parser.add_argument(
        "--out_dir",
        default="",
        type=str,
        help="Parent directory for saving this experiment results",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        help="Dataset to be used for training/testing",
    )
    parser.add_argument(
        "--seed_id", type=int, default=1, help="Setting seed for all our experiments"
    )

    # Training hyper-parameters
    parser.add_argument(
        "--model_type", type=str, default="ResNet", help="Type of task model."
    )
    parser.add_argument(
        "--model_depth",
        type=int,
        default=50,
        help="Depth of ResNet model. Not Supported for resnet_style.resnet_1",
    )
    parser.add_argument(
        "--train_batch_size", type=int, help="Batch Size used for training", default=256
    )
    parser.add_argument(
        "--test_batch_size", type=int, help="Batch Size used for training", default=200
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Task classifier learning rate"
    )
    parser.add_argument(
        "--wt_decay", type=float, default=1e-4, help="L2 regularization"
    )
    parser.add_argument(
        "--clf_epochs",
        type=int,
        default=1,
        help="Number of epochs used to train classifier",
    )
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument(
        "--eval_period",
        type=int,
        default=1,
        help="Period epoch where we want to evaluate model performance",
    )
    parser.add_argument(
        "--checkpoint_period",
        type=int,
        default=1,
        help="Epoch Period at which model snapshots are saved",
    )
    # parser.add_argument('--simple_augmentations',action='store_true',help="For turning on simple augmentations. Default is false")
    parser.add_argument(
        "--log_iter",
        type=int,
        default=10,
        help="Period after which train and val statistics are printed",
    )

    # Optimizer arguments
    parser.add_argument("--optim", type=str, default="adam", help="Type of optimizer")
    parser.add_argument(
        "--optim_mom",
        type=float,
        default=0.9,
        help="Momentum used only when optimizer is SGD",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="True for running SGD with nesterov. By default, False",
    )

    # Different data partitions
    parser.add_argument(
        "--lSetPath", type=str, default="", help="Path of lSet indices numpy array"
    )
    parser.add_argument(
        "--uSetPath", type=str, default="", help="Path of uSet indices numpy array"
    )
    parser.add_argument(
        "--valSetPath", type=str, default="", help="Path of valSet indices numpy array"
    )
    parser.add_argument(
        "--train_dir", type=str, default="", help="Path of training dataset dir"
    )
    parser.add_argument(
        "--test_dir", type=str, default="", help="Path of test dataset directory"
    )
    parser.add_argument(
        "--n_workers", type=int, default=4, help="Number of Data Loaders"
    )

    # Active learning arguments
    parser.add_argument(
        "--al_mode",
        action="store_true",
        help="For Activating active learning experiment. Default is false",
    )
    parser.add_argument(
        "--al_max_iter", type=int, default=1, help="Maximum active learning iterations"
    )
    parser.add_argument(
        "--sampling_fn",
        type=str,
        default="random",
        help="Sampling function for active sampling.",
    )
    parser.add_argument(
        "--budget_size",
        type=int,
        default=5000,
        help="Budget size for active learning iterations",
    )
    # parser.add_argument('--clf_model_path',type=str,default="",help="Path for loading classifier")
    parser.add_argument(
        "--init_partition",
        type=float,
        default=0.0,
        required=True,
        help="initial percent of data as lSet Size",
    )
    parser.add_argument(
        "--step_partition",
        type=float,
        default=10,
        help="partition percent by which we increase lSetSize",
    )
    parser.add_argument(
        "--uc_n_bins",
        type=int,
        default=0,
        help="Number of bins used by uncertainty principle",
    )
    parser.add_argument(
        "--dropout_iterations",
        type=int,
        default=0,
        help="Number of dropout fwd passes. Used by DBAL",
    )

    ##VAAL arguments --needed to run vaal or vaal[-d]
    parser.add_argument(
        "--vaal_z_dim", type=int, default=32, help="Latent code dimension of VAE"
    )
    parser.add_argument(
        "--vaal_vae_bs",
        type=int,
        default=64,
        help="Batch Size used for training VAAL sampling",
    )
    parser.add_argument(
        "--vaal_epochs",
        type=int,
        default=15,
        help="Number of epochs required to train VAAL sampling. Note that for VAAL sampling one epoch means a pass over lSet and uSet",
    )
    parser.add_argument(
        "--vaal_vae_lr",
        type=float,
        default=5e-4,
        help="Learning rate used for VAE model",
    )
    parser.add_argument(
        "--vaal_disc_lr",
        type=float,
        default=5e-4,
        help="Learning rate used for Discriminator model",
    )
    parser.add_argument(
        "--vaal_beta",
        type=float,
        default=1.0,
        help="Beta term focusing on KL term in VAE loss",
    )
    parser.add_argument(
        "--vaal_adv_param",
        type=float,
        default=1.0,
        help="Weight given to adversarial signal in VAAL sampling",
    )
    parser.add_argument(
        "--vaal_im_size",
        type=int,
        default=32,
        help="Image size used for training VAE (and Disc)",
    )

    # SWA arguments
    parser.add_argument(
        "--swa_mode", action="store_true", help="For activating swa mode"
    )
    parser.add_argument(
        "--swa_lr",
        type=float,
        default=5e-3,
        help="Learning rate used in swa optimization",
    )
    parser.add_argument(
        "--swa_freq",
        type=int,
        default=50,
        help="Frequency at which we use model snapshots",
    )
    parser.add_argument(
        "--swa_iter", type=int, default=50, help="Starting iteration in swa optimizer"
    )
    parser.add_argument(
        "--only_swa",
        action="store_true",
        help="Used to train only in SWA mode. Typically used post training.",
    )
    parser.add_argument(
        "--only_swa_partition",
        type=int,
        default=10,
        help="Specific partition from data to load",
    )
    parser.add_argument(
        "--swa_epochs", type=int, default=0, help="Number of epochs to train with SWA"
    )

    # Rand_Aug arguments
    parser.add_argument(
        "--device_ids",
        type=int,
        default=[0],
        nargs="+",
        help="List of Gpu Ids for running the code. For example: 0 2 or 1 3 4",
    )
    parser.add_argument(
        "--rand_aug",
        action="store_true",
        help="To activate randaug method. By default it is false",
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

    # Noisy arguments
    parser.add_argument(
        "--noisy_oracle", type=float, default=0.0, help="Oracle Noisy Percentage"
    )

    # Imbalanced exp arguments
    parser.add_argument(
        "--isimbalanced",
        action="store_true",
        help="Switch to turn on imabalanced experiments",
    )

    # Ensemble arguments
    parser.add_argument(
        "--ensemble", action="store_true", help="Switch to turn Ensemble Learning"
    )
    parser.add_argument(
        "--num_ensembles",
        type=int,
        default=3,
        help="Number of models to train in ensemble",
    )
    parser.add_argument(
        "--ens_epochs",
        type=int,
        default=1,
        help="Number of epochs used to train ensemble",
    )

    # Transferrability Experiments
    parser.add_argument(
        "--isTransferExp",
        action="store_true",
        help="Switch to turn on Transfer experiments",
    )
    parser.add_argument(
        "--transfer_dir_specific",
        type=str,
        default="vanilla",
        help="Directory specific for transfer experiments.",
    )
    parser.add_argument(
        "--transfer_model_type", type=str, default="ResNet", help="Type of task model."
    )
    parser.add_argument(
        "--transfer_model_style",
        type=str,
        default="ResNet",
        help="Style of task model.",
    )
    parser.add_argument(
        "--transfer_model_depth",
        type=int,
        default=50,
        help="Depth of ResNet model. Not Supported for resnet_style.resnet_1",
    )

    parser.add_argument(
        "--no_automl", action="store_false", help="Switch to turn off pruning"
    )

    # parser.add_argument('--transfer_data_split',type=int,default=50,help="Data split repository from where indexed sets will be used")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def get_al_args(args, data_splits, i, directory_specific, alStart):
    print(
        "~~ Constructing al_args for {} with alStart: {}".format(
            data_splits[i], alStart
        )
    )
    best_model_path = get_best_model_path(
        args, data_splits, i, directory_specific, alStart
    )

    if alStart:

        # temp_out_dir = '{}{}/{}/{}/{}/{}/{}/'.format(args.out_dir, args.dataset, str(data_splits[i]),str(args.seed_id), args.model_type+'_depth_'+str(args.model_depth), directory_specific, args.sampling_fn)
        temp_out_dir = get_path_directory(
            alStart, args, data_splits[i], directory_specific
        )

        al_args = [
            # 'ACTIVE_LEARNING.LSET_PATH', args.lSetPath if i==1 else args.out_dir+args.dataset+'/'+str(data_splits[i-1])+'/'+str(args.seed_id)+'/'+args.sampling_fn+'/lSet.npy', \
            # 'ACTIVE_LEARNING.USET_PATH', args.uSetPath if i==1 else args.out_dir+args.dataset+'/'+str(data_splits[i-1])+'/'+str(args.seed_id)+'/'+args.sampling_fn+'/uSet.npy', \
            "ACTIVE_LEARNING.LSET_PATH",
            args.lSetPath
            if i == 1
            else get_path_directory(
                alStart, args, data_splits[i - 1], directory_specific, i
            )
            + "/lSet.npy",
            "ACTIVE_LEARNING.USET_PATH",
            args.uSetPath
            if i == 1
            else get_path_directory(
                alStart, args, data_splits[i - 1], directory_specific, i
            )
            + "/uSet.npy",
            "ACTIVE_LEARNING.ACTIVATE",
            args.al_mode,
            "OUT_DIR",
            temp_out_dir,
            "ACTIVE_LEARNING.N_BINS",
            args.uc_n_bins,
            "ACTIVE_LEARNING.VALSET_PATH",
            args.valSetPath,
            "ACTIVE_LEARNING.MODEL_LOAD_DIR",
            best_model_path,
            "ACTIVE_LEARNING.BUDGET_SIZE",
            args.budget_size,
            "ACTIVE_LEARNING.SAMPLING_FN",
            args.sampling_fn,
            "ACTIVE_LEARNING.DROPOUT_ITERATIONS",
            args.dropout_iterations,
            "ACTIVE_LEARNING.NOISY_ORACLE",
            args.noisy_oracle,
            "ACTIVE_LEARNING.DATA_SPLIT",
            round(float(data_splits[i]), 1),
            "NUM_GPUS",
            args.n_GPU,
            "DIR_SPECIFIC",
            directory_specific,
        ]
    else:
        temp_out_dir = get_path_directory(
            alStart, args, data_splits[i], directory_specific, i=0
        )
        # temp_out_dir = '{}{}/{}/{}/{}/{}/{}/'.format(args.out_dir, args.dataset, str(data_splits[i]),str(args.seed_id), args.model_type+'_depth_'+str(args.model_depth),directory_specific, args.sampling_fn)
        al_args = [
            # 'ACTIVE_LEARNING.LSET_PATH', args.out_dir+args.dataset+'/'+str(data_splits[i-1])+"/"+str(args.seed_id)+"/"+args.model_type+'_depth_'+str(args.model_depth)+'/'+ directory_specific +'/'+args.sampling_fn+'/lSet.npy', \
            # 'ACTIVE_LEARNING.USET_PATH', args.out_dir+args.dataset+'/'+str(data_splits[i-1])+"/"+str(args.seed_id)+"/"+args.model_type+'_depth_'+str(args.model_depth)+ '/'+ directory_specific +'/'+args.sampling_fn+'/uSet.npy', \
            "ACTIVE_LEARNING.LSET_PATH",
            get_path_directory(alStart, args, data_splits[i - 1], directory_specific)
            + "/lSet.npy",
            "ACTIVE_LEARNING.USET_PATH",
            get_path_directory(alStart, args, data_splits[i - 1], directory_specific)
            + "/uSet.npy",
            "ACTIVE_LEARNING.ACTIVATE",
            args.al_mode,
            "OUT_DIR",
            temp_out_dir,
            "ACTIVE_LEARNING.N_BINS",
            args.uc_n_bins,
            "ACTIVE_LEARNING.VALSET_PATH",
            args.valSetPath,
            "ACTIVE_LEARNING.MODEL_LOAD_DIR",
            best_model_path,
            "ACTIVE_LEARNING.BUDGET_SIZE",
            args.budget_size,
            "ACTIVE_LEARNING.SAMPLING_FN",
            args.sampling_fn,
            "ACTIVE_LEARNING.DROPOUT_ITERATIONS",
            args.dropout_iterations,
            "ACTIVE_LEARNING.NOISY_ORACLE",
            args.noisy_oracle,
            "ACTIVE_LEARNING.DATA_SPLIT",
            round(float(data_splits[i]), 1),
            "NUM_GPUS",
            args.n_GPU,
            "DIR_SPECIFIC",
            directory_specific,
        ]
    print("======Inside get_al_args=====")
    print("al_args: {}".format(al_args))
    print("=============================")
    return al_args, temp_out_dir
