# GIL for tabular data with MLP model


## Data

Training/testing data should be stored, under the `data` folder, as numpy matrices with dimensions `[N, d]`, where `N` is the number of samples and `d` is the dimension of each sample.


## For  GIL

To train GIL on the training dataset stored under `data` folder:

`python tabular_mlp_GIL_train.py`

**[options]**

`-no_gpu`	`bool`	"Train w/o using GPUs"	`default=False`

`-gpu` 	`int` 	"Select which GPU to use" 	`default=0`

`-lr_prediction_model`	`float`	"Set learning rate for training the LSTM prediction model"	`default=0.001`

`-lr_actor`	`float`	"Set learning rate for training the actor"	`default=0.0001`

`-lr_critic`	`float`	"Set learning rate for training the critic"	`default=0.001`

`-decay_step`	`int`	"Set exponential decay step"	`default=750`

`-decay_rate`	`float`	"Set exponential decay rate"	`default=0.9`

`-decay_lr_actor`	`float`	"Set decay rate the learning rate of the actor"	`default=0.965`

`-decay_lr_critic`	`float`	"Set decay rate the learning rate of the critic"	`default=0.965`

`-training_steps`	`int`	"Set max number of training epochs"	`default=3000`

`-seed`	`int`	"Set random seed"	`default=2599`

`-exploration_prob`	`float`	"Initial probability of random exploration (p3 in Appendix D) in the behavioral policy"	`default=0.4`

`-heuristic_prob`	`float`	"Initial probability of following the heuristic (p2 in Appendix D) in the behavioral policy"	`default=0.5`

`-exploration_prob_decay`	`float`	"Rate of decaying the probability of random exploration in each step"	`default=0.999`

`-heuristic_prob_decay`	`float`	"Rate of decaying the probability of following the heuristic in each step"	`default=0.999`

`-replay_buffer`	`int`	"Size of experience replay buffer for training actor and critic"	`default=10**4`

----------------------------------------------------------------------------------------------------------------

To evaluate GIL on the testing dataset stored under `data` folder:

`python tabular_mlp_GIL_eval.py -ckpt_path <PATH_TO_CKPT_FILES>`

**[options]**

`-no_gpu`	`bool`	"Train w/o using GPUs"	`default=False`

`-gpu` 	`int` 	"Select which GPU to use" 	`default=0`

----------------------------------------------------------------------------------------------------------------

To load the checkpoint of GIL pre-trained using the 35% missing ophthalmic dataset and default parameters:

`python tabular_mlp_GIL_eval.py -ckpt_path ./saved_model/ophthalmic_35_missing_GIL_CKPT/`

----------------------------------------------------------------------------------------------------------------


## For  GIL-D


To train GIL-D on the training dataset stored under `data` folder:

`python tabular_mlp_GIL-D_train.py`

**[options]**

`-no_gpu`	`bool`	"Train w/o using GPUs"	`default=False`
`-gpu` 	`int` 	"Select which GPU to use" 	`default=0`
`-lr_prediction_model`	`float`	"Set learning rate for training the LSTM prediction model"	`default=0.001`
`-lr_actor`	`float`	"Set learning rate for training the actor"	`default=0.00005`
`-lr_critic`	`float`	"Set learning rate for training the critic"	`default=0.001`
`-decay_step`	`int`	"Set exponential decay step"	`default=750`
`-decay_rate`	`float`	"Set exponential decay rate"	`default=0.9`
`-decay_lr_actor`	`float`	"Set decay rate the learning rate of the actor"	`default=0.965`
`-decay_lr_critic`	`float`	"Set decay rate the learning rate of the critic"	`default=0.965`
`-training_steps`	`int`	"Set max number of training epochs"	`default=3000`
`-seed`	`int`	"Set random seed"	`default=2599`
`-exploration_prob`	`float`	"Initial probability of random exploration (p3 in Appendix D) in the behavioral policy"	`default=0.4`
`-heuristic_prob`	`float`	"Initial probability of following the heuristic (p2 in Appendix D) in the behavioral policy"	`default=0.5`
`-exploration_prob_decay`	`float`	"Rate of decaying the probability of random exploration in each step"	`default=0.999`
`-heuristic_prob_decay`	`float`	"Rate of decaying the probability of following the heuristic in each step"	`default=0.999`
`-replay_buffer`	`int`	"Size of experience replay buffer for training actor and critic"	`default=10**4`

----------------------------------------------------------------------------------------------------------------

To evaluate GIL-D on the testing dataset stored under `data` folder:

`python tabular_mlp_GIL-D_eval.py -ckpt_path <PATH_TO_CKPT_FILES>`

**[options]**

`-no_gpu`	`bool`	"Train w/o using GPUs"	`default=False`

`-gpu` 	`int` 	"Select which GPU to use" 	`default=0`

----------------------------------------------------------------------------------------------------------------

To load the checkpoint of GIL-D pre-trained using the 35% missing ophthalmic dataset and default parameters:

`python tabular_mlp_GIL-D_eval.py -ckpt_path ./saved_model/ophthalmic_35_missing_GIL-D_CKPT/`

----------------------------------------------------------------------------------------------------------------

