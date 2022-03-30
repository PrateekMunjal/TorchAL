# TorchAL codebase

Source code for CVPR 2022 Paper: [Towards Robust and Reproducible Active Learning Using Neural Networks](https://arxiv.org/abs/2002.09564)

<img src="https://raw.githubusercontent.com/PrateekMunjal/wowal/master/paper_images/cifar_five_lSet_statistics_latest_wo_cog_HR.png" alt="cifar_five_lSet_statistics_latest_wo_cog"/>

## What is TorchAL?

TL;DR: An Active Learning framework built on top of [pycls](https://github.com/facebookresearch/pycls).

TorchAL is an evaluation toolkit with a motive to advance the **reproducible** research in deep active learning. We currently implement state-of-the-art 
active learning (AL) algorithms. Our tookit extends the widely used [pycls](https://github.com/facebookresearch/pycls) codebase under AL settings.

## Features of TorchAL

* We report strong random baselines across widely used architectures and datasets.
* Our baselines are well-trained using AutoML which helps in reducing the bias introduced by choosing sub-optimal hyper-parameters.
* As we dream of reproducible results in AL, we release the training, validation index sets so that newer AL methods in future can use exact labeled set for training as we used to report our strong baselines.
* For familiarity with the codebase, we recommend interested users to read the tutorials. 

## AutoML in Active Learning

During AL iterations we observed that labeled set changes and therefore does the class distribution too. 
Thus in contrast to contemporary AL methods which fix the training hyper-parameters at the start of AL, we 
tune the training hyper-parameters using AutoML. To facilitate this we make use of [optuna](https://optuna.org/)
to perform random search over 50 trials for each AL cycle.

<img src="./paper_images/AL_cycles_anim.gif" />


## Requirements
* For creating a conda environment, kindly refer to [conda_env.yaml](https://github.com/PrateekMunjal/wowal/blob/master/conda_env.yml)
* For installing dependencies via pip, kindly refer to [requirements.txt](https://github.com/PrateekMunjal/wowal/blob/master/requirements.txt)

NOTE: In either case we have to download the dataset indexes and follow tools/train_al.py:
Dataset index sets
```shell
wget https://github.com/PrateekMunjal/wowal/blob/master/dataset_indexes.zip

```

## Installation

### From source

```
!wget codebase repo link
python setup.py install
```

### From pip
```
pip install torchal
```

## AL algorithms implemented

* Uncertainty
* Coreset
* BALD
* DBAL
* VAAL
* QBC
* Random Baseline

## Experimental Settings

NOTE: Below 5% meains 5 percent of the full training dataset. For example, for CIFAR10 - we have 50k datapoints as their official training split so with 5% we have 2500 datapoints in our initial labeled set.

* Different budget sizes: 5%, 10% 
* Different validation set sizes: 2%, 5%, 10%
* Effect of regularization techniques

## Examples

### Run the random baseline without strong regularization

<details>
  <summary>Expand to see the full script</summary>

```
pythonExec=$1

cd /raid/shadab/prateek/newcode

# script params
port=5035
sampling_fn=uncertainty
lSet_partition=1
base_seed=1
num_GPU=2
al_iterations=4 #7 #4
num_aml_trials=3 #50
budget_size=5000 #2500

dataset=CIFAR10
init_partition=10
step_partition=10
clf_epochs=5 #150
num_classes=10

log_iter=40

#Data arguments
train_dir=/raid/shadab/prateek/newcode/data/$dataset/train-$dataset/
test_dir=/raid/shadab/prateek/newcode/data/$dataset/test-$dataset/
lSetPath=/raid/shadab/prateek/newcode/data/$dataset/partition_$lSet_partition/lSet_$dataset.npy
uSetPath=/raid/shadab/prateek/newcode/data/$dataset/partition_$lSet_partition/uSet_$dataset.npy
valSetPath=/raid/shadab/prateek/newcode/data/$dataset/partition_$lSet_partition/valSet_$dataset.npy

#for lSet 1
out_dir=/raid/shadab/prateek/newcode/results 

# for other lSet Exps
# out_dir=/raid/shadab/prateek/newcode/results_lSetPartitions

#model_types: (i) wide_resnet_50 (ii) wide_resnet_28_10 (iii) wide_resnet_28_2

model_style=vgg_style
model_type=vgg #resnet_shake_shake
model_depth=16 #26

export CUDA_VISIBLE_DEVICES=0,1

$pythonExec tools/main_aml.py --n_GPU $num_GPU \
--port $port --sampling_fn $sampling_fn --lSet_partition $lSet_partition \
--seed_id $base_seed \
--init_partition $init_partition --step_partition $step_partition \
--dataset $dataset --budget_size $budget_size \
--out_dir $out_dir \
--num_aml_trials $num_aml_trials --num_classes $num_classes \
--al_max_iter $al_iterations \
--model_type $model_type --model_depth $model_depth \
--clf_epochs $clf_epochs \
--eval_period 1 --checkpoint_period 1 \
--lSetPath $lSetPath --uSetPath $uSetPath --valSetPath $valSetPath \
--train_dir $train_dir --test_dir $test_dir \
--dropout_iterations 25 \
--cfg configs/$dataset/$model_style/$model_type/R-18_4gpu_unreg.yaml \
--vaal_z_dim 32 --vaal_vae_bs 64 --vaal_epochs 15 \
--vaal_vae_lr 5e-4 --vaal_disc_lr 5e-4 --vaal_beta 1.0 --vaal_adv_param 1.0 \

```
</details>

<br>
Usage: Assume above script is named as **run.sh**, then we can simply run it 

```
sh run.sh `which python`
```

<br>

### Run the random baseline with strong regularization

In the above script we only need to add few more switches to add strong-regularization.

```
swa_lr=5e-4
swa_freq=50
swa_epochs=5 #50

...
--rand_aug --swa_mode --swa_freq $swa_freq --swa_lr $swa_lr \
--swa_epochs $swa_epochs --swa_iter 0 \

```

## Citing TorchAL

If you use TorchAL, please consider citing:

    @inproceedings{Munjal2022TorchAL,
        title={Towards Robust and Reproducible Active Learning Using Neural Networks}, 
        author={Prateek Munjal and Nasir Hayat and Munawar Hayat and Jamshid Sourati 
                and Shadab Khan},
        booktitle={CVPR},
        year={2022}
    }

## Acknowledgement to codebases which were referenced during the building of TorchAL

* [pycls](https://github.com/facebookresearch/pycls)
* [Coreset](https://github.com/ozansener/active_learning_coreset)
* [VAAL](https://github.com/sinhasam/vaal)
