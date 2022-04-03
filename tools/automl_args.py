import argparse

parser = argparse.ArgumentParser(description="Arguments for auto ml experiments")

parser.add_argument("--n_GPU", type=int, help="Number of GPUS. Default is 1", default=1)
parser.add_argument(
    "--port", type=int, help="Port for running ddp. Default is 10001", default=10001
)

parser.add_argument(
    "--lSet_partition",
    type=int,
    help="Partition number of lSet. Default is 1.",
    default=1,
)
# parser.add_argument('--start_id', type=int, help="Seed id. Default is 1.", default=1)
parser.add_argument(
    "--seed_id", type=int, default=1, help="Setting seed for all our experiments"
)

parser.add_argument(
    "--init_partition",
    type=float,
    help="Split percent of partition. Default is 10.0",
    default=10.0,
)
parser.add_argument(
    "--step_partition",
    type=float,
    help="Step partition to perform AL iterations. Default is 10.0",
    default=10.0,
)

parser.add_argument(
    "--dataset",
    type=str,
    help="Specify the dataset. Default is CIFAR10",
    default="CIFAR10",
)
parser.add_argument(
    "--out_dir", type=str, help="Directory for saving the results.", required=True
)
parser.add_argument(
    "--budget_size",
    type=int,
    help="Size of active set to be chosen for AL iteration",
    default=10,
)

parser.add_argument(
    "--num_aml_trials",
    type=int,
    help="Number of trials performed during automl",
    required=True,
)
parser.add_argument(
    "--clf_epochs",
    type=int,
    default=1,
    help="Number of epochs used to train classifier",
)
parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
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
parser.add_argument(
    "--log_iter",
    type=int,
    default=10,
    help="Period after which train and val statistics are printed",
)
parser.add_argument(
    "--al_max_iter", type=int, required=True, help="Maximum active learning iterations"
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
    "--rand_aug_M", type=int, default=5, help="Magnitude for augmentation operations"
)

parser.add_argument(
    "--swa_mode",
    action="store_true",
    help="To activate SWA method. By default it is false",
)
parser.add_argument(
    "--swa_lr", type=float, default=5e-3, help="Learning rate used in swa optimization"
)
parser.add_argument(
    "--swa_freq", type=int, default=50, help="Frequency at which we use model snapshots"
)
parser.add_argument(
    "--swa_iter", type=int, default=50, help="Starting iteration in swa optimizer"
)
parser.add_argument(
    "--swa_epochs", type=int, default=0, help="Number of epochs to train with SWA"
)

parser.add_argument("--model_type", type=str, help="Model type", required=True)
parser.add_argument("--model_depth", type=int, help="Depth of DL model", required=True)

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
    "--cfg", dest="cfg_file", help="Config file", required=True, type=str
)

# AL methods based args
parser.add_argument(
    "--sampling_fn",
    type=str,
    help="Name of Sampling function. Default is random.",
    default="random",
)
parser.add_argument(
    "--dropout_iterations",
    type=int,
    default=0,
    help="Number of dropout fwd passes. Used by DBAL/BALD",
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
    "--vaal_vae_lr", type=float, default=5e-4, help="Learning rate used for VAE model"
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

# Reuse AML Exps
parser.add_argument(
    "--reuse_aml", action="store_true", help="Switch to reuse automl experiment results"
)
parser.add_argument(
    "--reuse_aml_seed",
    type=int,
    default=1,
    help="For which automl experiment results exist",
)

# For imbalanced experiments
parser.add_argument(
    "--isimbalanced",
    action="store_true",
    help="Switch to turn on imabalanced experiments",
)

# Noisy arguments
parser.add_argument(
    "--noisy_oracle", type=float, default=0.0, help="Oracle Noisy Percentage"
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
    "--transfer_model_style", type=str, default="ResNet", help="Style of task model."
)
parser.add_argument(
    "--transfer_model_depth",
    type=int,
    default=50,
    help="Depth of ResNet model. Not Supported for resnet_style.resnet_1",
)

args = parser.parse_args()
