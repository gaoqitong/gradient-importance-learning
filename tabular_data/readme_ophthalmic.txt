************************
*****  For  GIL  *****
************************


To train GIL on the 35% missing ophthalmic training dataset:

python ophthalmic_mlp_GIL_train.py

[options]
-no_gpu	bool	"Train w/o using GPUs"	default=False
-gpu 	int 	"Select which GPU to use" 	default=0
-lr_prediction_model	float	"Set learning rate for training the LSTM prediction model"	default=0.001
-lr_actor	float	"Set learning rate for training the actor"	default=0.0001
-lr_critic	float	"Set learning rate for training the critic"	default=0.001
-decay_step	int	"Set exponential decay step"	default=750
-decay_rate	float	"Set exponential decay rate"	default=0.9
-decay_lr_actor	float	"Set decay rate the learning rate of the actor"	default=0.965
-decay_lr_critic	float	"Set decay rate the learning rate of the critic"	default=0.965
-training_steps	int	"Set max number of training epochs"	default=3000
-seed	int	"Set random seed"	default=2599
-exploration_prob	float	"Initial probability of random exploration (p3 in Appendix D) in the behavioral policy"	default=0.4
-heuristic_prob	float	"Initial probability of following the heuristic (p2 in Appendix D) in the behavioral policy"	default=0.5
-exploration_prob_decay	float	"Rate of decaying the probability of random exploration in each step"	default=0.999
-heuristic_prob_decay	float	"Rate of decaying the probability of following the heuristic in each step"	default=0.999
-replay_buffer	int	"Size of experience replay buffer for training actor and critic"	default=10**4

----------------------------------------------------------------------------------------------------------------

To evaluate GIL on the 35% missing ophthalmic testing dataset:

python ophthalmic_mlp_GIL_eval.py -ckpt_path <PATH_TO_CKPT_FILES>

[options]
-no_gpu	bool	"Train w/o using GPUs"	default=False
-gpu 	int 	"Select which GPU to use" 	default=0

----------------------------------------------------------------------------------------------------------------

To load the checkpoint of GIL pre-trained using the default parameters:

python ophthalmic_mlp_GIL_eval.py -ckpt_path ./saved_model/ophthalmic_35_missing_GIL_CKPT/

----------------------------------------------------------------------------------------------------------------

**************************
*****  For  GIL-D  *****
**************************


To train GIL-D on the 35% missing ophthalmic training dataset:

python ophthalmic_mlp_GIL-D_train.py

[options]
-no_gpu	bool	"Train w/o using GPUs"	default=False
-gpu 	int 	"Select which GPU to use" 	default=0
-lr_prediction_model	float	"Set learning rate for training the LSTM prediction model"	default=0.001
-lr_actor	float	"Set learning rate for training the actor"	default=0.00005
-lr_critic	float	"Set learning rate for training the critic"	default=0.001
-decay_step	int	"Set exponential decay step"	default=750
-decay_rate	float	"Set exponential decay rate"	default=0.9
-decay_lr_actor	float	"Set decay rate the learning rate of the actor"	default=0.965
-decay_lr_critic	float	"Set decay rate the learning rate of the critic"	default=0.965
-training_steps	int	"Set max number of training epochs"	default=3000
-seed	int	"Set random seed"	default=2599
-exploration_prob	float	"Initial probability of random exploration (p3 in Appendix D) in the behavioral policy"	default=0.4
-heuristic_prob	float	"Initial probability of following the heuristic (p2 in Appendix D) in the behavioral policy"	default=0.5
-exploration_prob_decay	float	"Rate of decaying the probability of random exploration in each step"	default=0.999
-heuristic_prob_decay	float	"Rate of decaying the probability of following the heuristic in each step"	default=0.999
-replay_buffer	int	"Size of experience replay buffer for training actor and critic"	default=10**4

----------------------------------------------------------------------------------------------------------------

To evaluate GIL-D on the 35% missing ophthalmic testing dataset:

python ophthalmic_mlp_GIL-D_eval.py -ckpt_path <PATH_TO_CKPT_FILES>

[options]
-no_gpu	bool	"Train w/o using GPUs"	default=False
-gpu 	int 	"Select which GPU to use" 	default=0

----------------------------------------------------------------------------------------------------------------

To load the checkpoint of GIL-D pre-trained using the default parameters:

python ophthalmic_mlp_GIL-D_eval.py -ckpt_path ./saved_model/ophthalmic_35_missing_GIL-D_CKPT/

----------------------------------------------------------------------------------------------------------------

**************************
*****  For  GIL-H  *****
**************************


To train GIL-H on the ophthalmic training dataset:

python ophthalmic_mlp_GIL-H_train.py

[options]
-no_gpu	bool	"Train w/o using GPUs"	default=False
-gpu 	int 	"Select which GPU to use" 	default=0
-lr_prediction_model	float	"Set learning rate for training the LSTM prediction model"	default=0.0005
-decay_step	int	"Set exponential decay step"	default=1000
-decay_rate	float	"Set exponential decay rate"	default=0.8
-training_steps	int	"Set max number of training epochs"	default=3000
-seed	int	"Set random seed"	default=2599

----------------------------------------------------------------------------------------------------------------

To evaluate GIL-H on the ophthalmic testing dataset:

python ophthalmic_mlp_GIL-H_eval.py -ckpt_path <PATH_TO_CKPT_FILES>

[options]
-no_gpu	bool	"Train w/o using GPUs"	default=False
-gpu 	int 	"Select which GPU to use" 	default=0

----------------------------------------------------------------------------------------------------------------








