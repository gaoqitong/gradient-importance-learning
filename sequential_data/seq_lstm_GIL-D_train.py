import argparse
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os 
import multiprocessing as mp
from qnetwork import *
from utils import *
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
import scipy.stats as stats
import random

rnn = tf.contrib.rnn
slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-lstm_hidden_size", type=int, help="Set the size of LSTM hidden states DEFAULT=1024", default=1024)
parser.add_argument("-lr_prediction_model", type=float, help="Set learning rate for training the LSTM prediction model DEFAULT=0.0005", default=0.0005)
parser.add_argument("-lr_actor", type=float, help="Set learning rate for training the actor DEFAULT=0.0005", default=0.0005)
parser.add_argument("-lr_critic", type=float, help="Set learning rate for training the critic DEFAULT=0.0001", default=0.0001)
parser.add_argument("-decay_step", type=int, help="Set exponential decay step DEFAULT=500", default=500)
parser.add_argument("-decay_rate", type=float, help="Set exponential decay rate DEFAULT=1.0", default=1.0)
parser.add_argument("-decay_lr_actor", type=float, help="Set decay rate the learning rate of the actor DEFAULT=0.965", default=0.965)
parser.add_argument("-decay_lr_critic", type=float, help="Set decay rate the learning rate of the critic DEFAULT=0.965", default=0.965)
parser.add_argument("-training_steps", type=int, help="Set max number of training epochs DEFAULT=2000", default=2000)
parser.add_argument("-seed", type=int, help="Set random seed", default=2599)
parser.add_argument("-exploration_prob", type=float, help="Initial probability of random exploration (p3 in Appendix D) in the behavioral policy", default=0.6)
parser.add_argument("-heuristic_prob", type=float, help="Initial probability of following the heuristic (p2 in Appendix D) in the behavioral policy", default=0.15)
parser.add_argument("-exploration_prob_decay", type=float, help="Rate of decaying the probability of random exploration in each step", default=0.95)
parser.add_argument("-heuristic_prob_decay", type=float, help="Rate of decaying the probability of following the heuristic in each step", default=0.95)
parser.add_argument("-replay_buffer", type=int, help="Size of experience replay buffer for training actor and critic DEFAULT=10**4.", default=10**4)


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
        session_config = tf.ConfigProto(log_device_placement=False)
        session_config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        session_config = tf.ConfigProto(log_device_placement=False)
    SEED = args.seed
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    random.seed(SEED)

    if not os.path.exists("./saved_model"):
            os.mkdir("./saved_model")
    if not os.path.exists("./stats"):
            os.mkdir("./stats")
    if not os.path.exists("./stats/rl_log"):
            os.mkdir("./stats/rl_log")

    df_shock_train = pd.read_csv("./data/df_shock_train.csv", index_col="TrainSampleIdx")
    df_shock_test = pd.read_csv("./data/df_shock_test.csv", index_col="TrainSampleIdx")
    df_non_shock_train = pd.read_csv("./data/df_non_shock_train.csv", index_col="TrainSampleIdx")
    df_non_shock_test = pd.read_csv("./data/df_non_shock_test.csv", index_col="TrainSampleIdx")

    # determine a numerical value to represent nan values
    _max = -np.infty
    _min = np.infty
    for _df in [df_shock_train, df_non_shock_train]:
        _df_values = np.copy(_df.values)
        _df_values[np.isnan(_df.values)] = 0.
        if np.max(_df_values) > _max:
            _max = np.max(_df_values)
        if np.min(_df_values) < _min:
            _min = np.min(_df_values)

    nan_replacement = 3*_max
    # nan_replacement = 0.

    # determine the max sequence length
    max_seq_len = -np.infty
    for _df in [df_shock_train, df_non_shock_train, df_shock_test, df_non_shock_test]:
        max_for_current_df = np.max(np.unique(_df.index.values, return_counts=True)[1])
        if max_for_current_df > max_seq_len:
            max_seq_len = max_for_current_df


    # replace nan values
    for _df in [df_shock_train, df_non_shock_train, df_shock_test, df_non_shock_test]:
        _df[_df.isna()]=nan_replacement

    def seq_length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    # Re-formulate the training data such that each time half of the batch contains inputs with label 0 and the other half with label 1
    all_outs = []
    all_labels = []
    all_masks = []

    for idxs in np.asarray(np.split(np.concatenate([df_shock_train.index.unique() for i in range(64)]), len(df_shock_train.index.unique()))):
        outs = []

        for i in idxs:
            current_df = df_shock_train.loc[i]
            if isinstance(current_df, pd.core.frame.DataFrame):
                current_values = df_shock_train.loc[i].values
                out = np.vstack([current_values, np.zeros((max_seq_len-current_values.shape[0], current_values.shape[1]))])
            outs += [out]


        for i in idxs:
            current_df = df_non_shock_train.loc[i]
            if isinstance(current_df, pd.core.frame.DataFrame):
                current_values = df_non_shock_train.loc[i].values
                out = np.vstack([current_values, np.zeros((max_seq_len-current_values.shape[0], current_values.shape[1]))])
            outs += [out]

        outs = np.asarray(outs)
        masks = outs == nan_replacement
        masks = masks.astype(np.int32)
        labels = np.asarray([np.array([0.,1.]) for i in range(64)] + [np.array([1.,0.]) for i in range(64)])

        all_outs += [outs]
        all_labels += [labels]
        all_masks += [masks]

    def gen_train():
    # Output mask's dimensions correspond to [num_timesteps, batch_size, num_input/sequence_length]
        for i in range(len(all_outs)):
            yield all_outs[i], all_labels[i], all_masks[i]

    def gen_test():
        # Output mask's dimensions correspond to [num_timesteps, batch_size, num_input/sequence_length]
        for i in df_shock_test.index.unique():
            current_df = df_shock_test.loc[i]
            if isinstance(current_df, pd.core.frame.DataFrame):
                current_values = df_shock_test.loc[i].values
                out = np.vstack([current_values, np.zeros((max_seq_len-current_values.shape[0], current_values.shape[1]))])
                mask = out == nan_replacement
                mask = mask.astype(np.int32)
                label = np.array([0., 1.])
                yield out, label, mask
        for i in df_non_shock_test.index.unique():
            current_df = df_non_shock_test.loc[i]
            if isinstance(current_df, pd.core.frame.DataFrame):
                current_values = df_non_shock_test.loc[i].values
                out = np.vstack([current_values, np.zeros((max_seq_len-current_values.shape[0], current_values.shape[1]))])
                mask = out == nan_replacement
                mask = mask.astype(np.int32)
                label = np.array([1., 0.])
                yield out, label, mask


    # Setting up the truncated normal distribution for exploration

    lower, upper = 0, 1
    mu, sigma = 0, 0.2
    left_truncnorm = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    right_truncnorm = stats.truncnorm(
        (lower - 1.) / sigma, (upper - 1.) / sigma, loc=1., scale=sigma)

    # fig, ax = plt.subplots(1, sharex=True)
    # ax.hist(np.concatenate([left_truncnorm.rvs(10000),right_truncnorm.rvs(10000)]), normed=True)

    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    random.seed(SEED)

    # RL learning rates
    actor_lr, critic_lr = args.lr_actor, args.lr_critic

    # Prediction Model Parameters
    start_learning_rate = args.lr_prediction_model
    decay_step = args.decay_step
    decay_rate = args.decay_rate
    num_hidden = args.lstm_hidden_size

    # Threshold for decaying RL learning rates
    rl_reward_thres_for_decay = 5

    training_steps = args.training_steps
    batch_size = 128 # must be a multiple of 4

    num_input = df_shock_train.values.shape[1] 
    timesteps = max_seq_len # timesteps
    num_classes = 2 

    display_step = 10

    gpu = 0

    graph = tf.Graph()

    file_appendix = "SEQ_LSTM_GIL-D_" + str(start_learning_rate) + "_" + str(decay_step) + "_" + str(decay_rate) + "_" + str(num_hidden) + "_" + str(actor_lr) + "_" + str(critic_lr)


    def build_net(x, is_training=True, reuse=tf.AUTO_REUSE, graph=graph):

            with graph.as_default():
                with tf.variable_scope("lstm", reuse=reuse) as scope:
                    # LSTM Encoder
                    seq_len = seq_length(x)
                    enumerated_last_idxs = tf.cast(tf.stack([seq_len-1, tf.range(tf.shape(seq_len)[0])], axis=1), tf.int32)
                    x = tf.unstack(x, timesteps, 1)
                    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, reuse=reuse)
                    outputs, state, all_states = my_static_rnn(lstm_cell, x, dtype=tf.float32)
                    last_outputs = tf.gather_nd(outputs, enumerated_last_idxs)
                    # Output Layer
                    with slim.arg_scope([slim.fully_connected], 
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.random_uniform_initializer(0.001, 0.01),
                                            weights_regularizer=slim.l2_regularizer(0.005),
                                            biases_regularizer=slim.l2_regularizer(0.005),
                                            normalizer_fn = slim.batch_norm,
                                            normalizer_params = {"is_training": is_training},
                                            reuse = reuse,
                                            scope = scope):

                        logits = slim.fully_connected(last_outputs,num_classes,activation_fn=None, weights_regularizer=None, normalizer_fn=None, scope='logits')
                        pred = slim.softmax(logits, scope='pred')

                        return logits, pred, outputs, x, all_states, seq_len, last_outputs


    with graph.as_default():

        dataset_train = tf.data.Dataset.from_generator(gen_train, (tf.float32, tf.float32, tf.int32), ([batch_size, timesteps, df_shock_train.values.shape[1]],[batch_size, 2],[batch_size, timesteps, df_shock_train.values.shape[1]])).repeat(5).shuffle(10)
        input_train, label_train, mask_train = dataset_train.make_one_shot_iterator().get_next()

        dataset_test = tf.data.Dataset.from_generator(gen_test, (tf.float32, tf.float32, tf.int32), ([ timesteps, df_shock_train.values.shape[1]],[ 2],[timesteps, df_shock_train.values.shape[1]])).repeat(10000).batch(len(df_shock_test.index.unique())+len(df_non_shock_test.index.unique()))
        input_test, label_test, mask_test = dataset_test.make_one_shot_iterator().get_next()

        input_train_holder = tf.placeholder(shape=[batch_size, timesteps, num_input], dtype=tf.float32)
        label_train_holder = tf.placeholder(shape=[batch_size, 2], dtype=tf.float32)
        mask_train_holder = tf.placeholder(shape=[batch_size, timesteps, num_input], dtype=tf.int32)

        logits, prediction, outs, xs, states, seq_lens, last_outputs = build_net(input_train_holder)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_train_holder) + tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="lstm")), axis=0)
        learning_rate = tf.train.exponential_decay(start_learning_rate, tf.train.get_or_create_global_step(), decay_steps=decay_step, decay_rate=decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        missing_idxs = tf.where_v2(mask_train)
        missing_idxs = tf.stack([missing_idxs[:,1], missing_idxs[:,0], missing_idxs[:,2]], axis=-1)

        # Get the encoding LSTM weights and obtain the gradients using regular SGD sovler

        i_gates = [graph.get_tensor_by_name("lstm/rnn/basic_lstm_cell/split_"+str(t)+":0") if t>0 else graph.get_tensor_by_name("lstm/rnn/basic_lstm_cell/split:0") for t in range(timesteps)]
        j_gates = [graph.get_tensor_by_name("lstm/rnn/basic_lstm_cell/split_"+str(t)+":1") if t>0 else graph.get_tensor_by_name("lstm/rnn/basic_lstm_cell/split:1") for t in range(timesteps)]
        f_gates = [graph.get_tensor_by_name("lstm/rnn/basic_lstm_cell/split_"+str(t)+":2") if t>0 else graph.get_tensor_by_name("lstm/rnn/basic_lstm_cell/split:2") for t in range(timesteps)]
        o_gates = [graph.get_tensor_by_name("lstm/rnn/basic_lstm_cell/split_"+str(t)+":3") if t>0 else graph.get_tensor_by_name("lstm/rnn/basic_lstm_cell/split:3") for t in range(timesteps)]

        grads_i = optimizer.compute_gradients(loss_op,i_gates)
        grads_i = [g[0] for g in grads_i]
        grads_j = optimizer.compute_gradients(loss_op,j_gates)
        grads_j = [g[0] for g in grads_j]
        grads_f = optimizer.compute_gradients(loss_op,f_gates)
        grads_f = [g[0] for g in grads_f]
        grads_o = optimizer.compute_gradients(loss_op,o_gates)
        grads_o = [g[0] for g in grads_o]

        grads_i_j_f_o = [tf.concat([grads_i[t], grads_j[t], grads_f[t], grads_o[t]], axis=1) for t in range(timesteps)]

        # Apply importance to the gradients calculated from regular SGD solver

        grad_importance = tf.placeholder(shape=[timesteps, batch_size, num_input], dtype=tf.float32)
        xs_for_grads = tf.multiply(xs, grad_importance)
        W_grads = tf.tensordot(xs_for_grads, grads_i_j_f_o, axes=[[0,1],[0,1]])/batch_size

        enumerated_seq_lens = tf.cast(tf.stack([seq_lens, tf.range(tf.shape(seq_lens)[0])], axis=1), tf.int32)

        def cond(i, e, o):
            return i < batch_size
        def body(i, e, o):
            o = tf.concat([o,tf.stack([tf.range(e[i,0]),tf.repeat(e[i,1],e[i,0])],axis=-1)],axis=0)
            return i+1, e, o

        _,_,nonzero_out_idxs = tf.while_loop(cond,body,[tf.constant(1, dtype=tf.int32), enumerated_seq_lens, tf.stack([tf.range(enumerated_seq_lens[0,0]),tf.repeat(enumerated_seq_lens[0,1],enumerated_seq_lens[0,0])],axis=-1)], shape_invariants=[tf.TensorShape([]),tf.TensorShape([None,2]),tf.TensorShape([None,2])])

        outs_non_zero = tf.gather_nd(outs,nonzero_out_idxs)
        outs_updates = tf.scatter_nd(indices=nonzero_out_idxs, updates=outs_non_zero, shape=[timesteps, batch_size, num_hidden])
        outs = tf.zeros((timesteps,batch_size,num_hidden)) + outs_updates
        U_grads = tf.tensordot(outs, grads_i_j_f_o, axes=[[0,1],[0,1]])/batch_size
        lstm_kernel_grads = tf.concat([W_grads,U_grads],axis=0)     

        logits_final, pred_final, _, _, _, _, _ = build_net(input_test, is_training=False)


        grads = optimizer.compute_gradients(loss_op, [v for v in tf.trainable_variables() if v.name.find("lstm")!=-1])
        grads = [g[0] for g in grads]

        grads[0] = lstm_kernel_grads


        grads_update_op = optimizer.apply_gradients(zip(grads, [v for v in tf.trainable_variables() if v.name.find("lstm")!=-1]))

        # Setting up metrics

        train_correct_pred = tf.equal(tf.cast(tf.argmax(prediction, 1),tf.float32), tf.cast(tf.argmax(label_train_holder, 1),tf.float32) )
        train_accuracy = tf.reduce_mean(tf.cast(train_correct_pred, tf.float32))
        train_kld = tf.keras.losses.KLDivergence()(prediction, label_train_holder)

        final_correct_pred = tf.equal(tf.cast(tf.argmax(pred_final, 1), tf.float32), tf.cast(tf.argmax(label_test, 1),tf.float32))
        final_accuracy = tf.reduce_mean(tf.cast(final_correct_pred, tf.float32))
        final_kld = tf.keras.losses.KLDivergence()(pred_final, label_test)

        final_score = pred_final[:,1]

        max_final_acc = tf.Variable(0, dtype=tf.float32, name="max_final_acc", trainable=False)
        assign_max_final_acc = max_final_acc.assign(final_accuracy)

    with graph.as_default():
        actor = Actor(graph=graph, state_dim=num_input*2+num_hidden*2, action_dim=num_input, learning_rate=actor_lr, tau=0.001, batch_size=batch_size, save_path="./saved_model/"+file_appendix+"/actor.ckpt")
        critic = Critic(graph=graph, state_dim=num_input*2+num_hidden*2, action_dim=num_input, learning_rate=critic_lr, tau=0.001, gamma=0.99, save_path="./saved_model/"+file_appendix+"/critic.ckpt")
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


    # Start training
    with tf.Session(config=session_config, graph=graph) as sess:
        sess.run(init)

        # Probability of random exploration (p3 in Appendix D) in the behavioral policy
        ## This probability will be decayed exponentially during training
        EXPLORATION_RATE = args.exploration_prob

        # Probability of following the heuristic (p2 in Appendix D) in the behavioral policy
        ## This probability will be decayed exponentially during training
        GUIDE_RATE = args.heuristic_prob

        ep_reward = 0
        ep_ave_max_q = 0

        data_in, label_in, s_mask = sess.run([input_train, label_train, mask_train])


        s_1, s_2 = sess.run([states, outs], feed_dict = {input_train_holder:data_in, label_train_holder:label_in, mask_train_holder:s_mask})
        s = np.concatenate([np.asarray(np.split(data_in,timesteps,axis=1)).reshape(timesteps,batch_size,num_input),
                       np.asarray(np.split(s_mask,timesteps,axis=1)).reshape(timesteps,batch_size,num_input)
                            ,s_1,s_2], axis=-1)


        reward_list = []
        ave_max_q_list = []
        replay_buffer = ReplayBuffer(args.replay_buffer, random_seed=SEED)

        # Run the initializer

        max_auc = 0.
        max_ap = 0.

        actor.update_target_network(sess)
        critic.update_target_network(sess)

        for step in range(training_steps):
            rand_num = np.random.rand(1)

            if rand_num <= EXPLORATION_RATE:
                a = np.concatenate([left_truncnorm.rvs(int(timesteps*batch_size*num_input/2)),right_truncnorm.rvs(int(timesteps*batch_size*num_input/2))])
                np.random.shuffle(a)
                a = a.reshape(timesteps, batch_size, num_input).astype(np.float32)

            elif rand_num <= GUIDE_RATE+EXPLORATION_RATE and rand_num > EXPLORATION_RATE:
                a = np.asarray(np.split((1-s_mask).astype(np.float32), timesteps, axis=1)).reshape(timesteps,batch_size,num_input)

            else:
                a = actor.predict(s.reshape(-1,num_input*2+num_hidden*2), sess)
                a = a.reshape(timesteps, batch_size, num_input)

            last_outs, _, kld = sess.run([last_outputs, grads_update_op, train_kld], feed_dict={grad_importance:a, input_train_holder:data_in, label_train_holder:label_in, mask_train_holder:s_mask})
            acc, score = sess.run([final_accuracy, final_score])
            data_in, label_in, s2_mask = sess.run([input_train, label_train, mask_train])
            s2_1, s2_2 = sess.run([states, outs], feed_dict = {input_train_holder:data_in, label_train_holder:label_in})
            s2 = np.concatenate([np.asarray(np.split(data_in,timesteps,axis=1)).reshape(timesteps,batch_size,num_input),
                       np.asarray(np.split(s_mask,timesteps,axis=1)).reshape(timesteps,batch_size,num_input)
                            ,s2_1,s2_2], axis=-1)
            r = np.repeat(-kld, batch_size)
            r_mse = mean_squared_error(last_outs[:batch_size//4, :], last_outs[batch_size//2:batch_size*3//4, :]) + \
                    mean_squared_error(last_outs[batch_size//4:batch_size//2, :], last_outs[batch_size*3//4:, :]) - \
                    mean_squared_error(last_outs[:batch_size//4, :],last_outs[batch_size//4:batch_size//2, :]) - \
                    mean_squared_error(last_outs[batch_size//2:batch_size*3//4, :],last_outs[batch_size*3//4:, :])
            r = r + 5*r_mse
            replay_buffer.add_batch([list(i) for i in zip(s.reshape(-1,num_input*2+num_hidden*2),a.reshape(-1,num_input),r,s2.reshape(-1,num_input*2+num_hidden*2))])

            if replay_buffer.size() > batch_size:
                s_batch, a_batch, r_batch, s2_batch = replay_buffer.sample_batch(batch_size)

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch, sess), sess)

                y_i = []
                for k in range(batch_size):
                    y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (batch_size, 1)), step, sess)

                ave_max_q = np.amax(predicted_q_value)
                ave_max_q_list += [ave_max_q]

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch, sess)
                grads = critic.action_gradients(s_batch, a_outs, sess)
                actor.train(s_batch, grads[0], step, sess)

                # Update target networks
                actor.update_target_network(sess)
                critic.update_target_network(sess)

            s = s2
            s_mask = s2_mask

            reward_list += [r[0]]

            # Decay p2 and p3 in the behavioral policy
            EXPLORATION_RATE = EXPLORATION_RATE * args.exploration_prob_decay
            GUIDE_RATE = GUIDE_RATE * args.heuristic_prob_decay


            if step % display_step == 0 and step > 0:
                # Calculate batch loss and accuracy
                loss, acc, train_acc = sess.run([loss_op, final_accuracy, train_accuracy], feed_dict = {input_train_holder:data_in, label_train_holder:label_in})
                auc = roc_auc_score(np.argmax(sess.run(label_test), axis=1), final_score.eval())
                ap = average_precision_score(np.argmax(sess.run(label_test), axis=1), final_score.eval())
                if np.mean(reward_list[-display_step:]) >= rl_reward_thres_for_decay:
                    actor.decay_learning_rate(args.decay_lr_actor, sess)
                    critic.decay_learning_rate(args.decay_lr_critic, sess)
                if acc > max_final_acc.eval():
                    max_auc = auc
                    max_ap = ap
                    sess.run(assign_max_final_acc)
                    saver.save(sess, "./saved_model/"+file_appendix+"/best.ckpt")
                print ("Step " + str(step) + ", Reward=" + str(np.sum(reward_list[-display_step:])) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(train_acc) + \
                      ", Max Testing Accuracy= ", "{:6f}".format(max_final_acc.eval()) + \
                      ", Max AUC= ", "{:6f}".format(max_auc) + \
                      ", Max AP= ", "{:6f}".format(max_ap) + \
                      ", Max Q= ", "{:6f}".format(np.mean(ave_max_q_list[-display_step:])))
                with open("./stats/rl_log/" + file_appendix + ".txt", "a") as myfile:
                    myfile.write("Step " + str(step) + ", Reward=" + str(np.sum(reward_list[-display_step:])) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(train_acc) + ", Max Final Accuracy= " + "{:6f}".format(max_final_acc.eval()) + ", Max AUC= " + "{:6f}".format(max_auc) + ", Max AP= " + "{:6f}".format(max_ap) + "\n")

        print ("Optimization Finished!")

        print ("Testing Accuracy:", sess.run(max_final_acc))
        print ("Testing AUC:", max_auc)

        # Record the hyper-parameters tried along with their performances
        with open("./stats/SEQ_LSTM_GIL-D.txt", "a") as myfile:
            myfile.write("%.6f\t%i\t%.3f\t%.6f\t%.6f\t%i\t%.6f\t%.6f\t%.6f\n" %(start_learning_rate, decay_step, decay_rate, actor_lr, critic_lr, num_hidden, max_final_acc.eval(), max_auc, max_ap))






















