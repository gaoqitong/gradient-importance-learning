# GIL for multivariate sequential data with LSTM model


## Data


The data used for training/testing (stored under `data` folder) is processed, denomynized and normalized upon the original dataset from https://physionet.org/content/mimiciii/1.4/. Download and use of this original data should follow The PhysioNet Credentialed Health Data License (https://physionet.org/content/mimiciii/view-license/1.4/). Herein we only provide the testing data for reproducibility check. The training data might be available upon request (contact qitong.gao@duke.edu).

**To ensure both training and evaluation scripts can be executed, the training data csv's under the `data` folder is filled with all zero's.**

The training/testing csv's are formulated as below. Each row is indexed by the patient ID and should contain records of all attributes for a single timestep for a single patient. For each patient, the recordings over multiple timesteps should follow increasing order over the time axis. **No** spaces should to be placed in between patients. 

|                      | Attr. #1  | Attr. #2 | Attr. #3 | ... |
|----------------------|-----------|----------|----------|-----|
| Patient ID #1 (t=0)  |           |          |          |     |
| Patient ID #1 (t=1)  |           |          |          |     |
| Patient ID #1 (t=2)  |           |          |          |     |
| ...                  |           |          |          |     |
| Patient ID #1 (t=T)  |           |          |          |     |
| Patient ID #2 (t=0)  |           |          |          |     |
| ...                  |           |          |          |     |
| Patient ID #2 (t=T)  |           |          |          |     |
| Patient ID #3 (t=0)  |           |          |          |     |
| ...                  |           |          |          |     |


## Run GIL


To train GIL on the training dataset stored under `data` folder:

`python seq_lstm_GIL_train.py`


**[options]**

`-no_gpu`	`bool`	"Train w/o using GPUs"	`default=False`

`-gpu` 	`int` 	"Select which GPU to use" 	`default=0`

`-lstm_hidden_size`	`int`	"Set the size of LSTM hidden states"	`default=1024`

`-lr_prediction_model`	float	"Set learning rate for training the LSTM prediction model"	`default=0.0005`

`-lr_actor`	`float`	"Set learning rate for training the actor"	`default=0.0005`

`-lr_critic`	`float`	"Set learning rate for training the critic"	`default=0.0001`

`-decay_step`	`int`	"Set exponential decay step"	`default=500`

`-decay_rate`	`float`	"Set exponential decay rate"	`default=1.0`

`-decay_lr_actor`	`float`	"Set decay rate the learning rate of the actor"	`default=0.965`

`-decay_lr_critic`	`float`	"Set decay rate the learning rate of the critic"	`default=0.965`

`-training_steps`	`int`	"Set max number of training epochs"	`default=2000`

`-seed`	`int`	"Set random seed"	`default=2599`

`-exploration_prob`	`float`	"Initial probability of random exploration (p3 in Appendix D) in the behavioral policy"	`default=0.6`

`-heuristic_prob`	`float`	"Initial probability of following the heuristic (p2 in Appendix D) in the behavioral policy"	`default=0.15`

`-exploration_prob_decay`	`float`	"Rate of decaying the probability of random exploration in each step"	`default=0.95`

`-heuristic_prob_decay`	`float`	"Rate of decaying the probability of following the heuristic in each step"	`default=0.95`

`-replay_buffer`	`int`	"Size of experience replay buffer for training actor and critic. Default to `10**5` but can be reduced to `10**4` if training too slow or occupies too much RAM."	`default=10**5`


----------------------------------------------------------------------------------------------------------------

To evaluate GIL on the testing dataset stored under `data` folder:

`python seq_lstm_GIL_eval.py -ckpt_path <PATH_TO_CKPT_FILES>`

**[options]**

`-no_gpu`	`bool`	"Train w/o using GPUs"	`default=False`

`-gpu` 	`int` 	"Select which GPU to use" 	`default=0`

----------------------------------------------------------------------------------------------------------------

To load the checkpoint of GIL pre-trained using the default parameters:

`python seq_lstm_GIL_eval.py -ckpt_path ./saved_model/MIMIC_LSTM_GIL_CKPT/`

----------------------------------------------------------------------------------------------------------------

## For  GIL-D

To train GIL-D on the training dataset stored under `data` folder:

`python seq_lstm_GIL-D_train.py`

**[options]**

`-no_gpu`	`bool`	"Train w/o using GPUs"	`default=False`

`-gpu` 	`int` 	"Select which GPU to use" 	`default=0`

`-lstm_hidden_size`	`int`	"Set the size of LSTM hidden states"	`default=1024`

`-lr_prediction_model`	`float`	"Set learning rate for training the LSTM prediction model"	`default=0.0005`

`-lr_actor`	`float`	"Set learning rate for training the actor"	`default=0.0005`

`-lr_critic`	`float`	"Set learning rate for training the critic"	`default=0.0001`

`-decay_step`	`int`	"Set exponential decay step"	`default=500`

`-decay_rate`	`float`	"Set exponential decay rate"	`default=1.0`

`-decay_lr_actor`	`float`	"Set decay rate the learning rate of the actor"	`default=0.965`

`-decay_lr_critic`	`float`	"Set decay rate the learning rate of the critic"	`default=0.965`

`-training_steps`	`int`	"Set max number of training epochs"	`default=2000`

`-seed`	`int`	"Set random seed"	`default=2599`

`-exploration_prob`	`float`	"Initial probability of random exploration (p3 in Appendix D) in the behavioral policy"	`default=0.6`

`-heuristic_prob`	`float`	"Initial probability of following the heuristic (p2 in Appendix D) in the behavioral policy"	`default=0.15`

`-exploration_prob_decay`	`float`	"Rate of decaying the probability of random exploration in each step"	`default=0.95`

`-heuristic_prob_decay`	`float`	"Rate of decaying the probability of following the heuristic in each step"	`default=0.95`

`-replay_buffer`	`int`	"Size of experience replay buffer for training actor and critic"	`default=10**4`

----------------------------------------------------------------------------------------------------------------

To evaluate GIL-D on the mimic testing dataset:

`python seq_lstm_GIL-D_eval.py -ckpt_path <PATH_TO_CKPT_FILES>`

**[options]**

`-no_gpu`	`bool`	"Train w/o using GPUs"	`default=False`

`-gpu` 	`int` 	"Select which GPU to use" 	`default=0`

----------------------------------------------------------------------------------------------------------------

To load the checkpoint of GIL-D pre-trained using the default parameters:

`python seq_lstm_GIL-D_eval.py -ckpt_path ./saved_model/MIMIC_LSTM_GIL-D_CKPT/`

----------------------------------------------------------------------------------------------------------------

