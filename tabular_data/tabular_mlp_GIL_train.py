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
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.stats as stats
import random

rnn = tf.contrib.rnn
slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument("-no_gpu", dest='no_gpu', action='store_true', help="Train w/o using GPUs")
parser.add_argument("-gpu", "--gpu_idx", type=int, help="Select which GPU to use DEFAULT=0", default=0)
parser.add_argument("-lr_prediction_model", type=float, help="Set learning rate for training the MLP prediction model DEFAULT=0.001", default=0.001)
parser.add_argument("-lr_actor", type=float, help="Set learning rate for training the actor DEFAULT=0.0001", default=0.0001)
parser.add_argument("-lr_critic", type=float, help="Set learning rate for training the critic DEFAULT=0.001", default=0.001)
parser.add_argument("-decay_step", type=int, help="Set exponential decay step DEFAULT=750", default=750)
parser.add_argument("-decay_rate", type=float, help="Set exponential decay rate DEFAULT=1.0", default=0.9)
parser.add_argument("-decay_lr_actor", type=float, help="Set decay rate the learning rate of the actor DEFAULT=0.965", default=0.965)
parser.add_argument("-decay_lr_critic", type=float, help="Set decay rate the learning rate of the critic DEFAULT=0.965", default=0.965)
parser.add_argument("-training_steps", type=int, help="Set max number of training epochs DEFAULT=3000", default=3000)
parser.add_argument("-seed", type=int, help="Set random seed", default=2599)
parser.add_argument("-exploration_prob", type=float, help="Initial probability of random exploration (p3 in Appendix D) in the behavioral policy", default=0.4)
parser.add_argument("-heuristic_prob", type=float, help="Initial probability of following the heuristic (p2 in Appendix D) in the behavioral policy", default=0.5)
parser.add_argument("-exploration_prob_decay", type=float, help="Rate of decaying the probability of random exploration in each step", default=0.999)
parser.add_argument("-heuristic_prob_decay", type=float, help="Rate of decaying the probability of following the heuristic in each step", default=0.999)
parser.add_argument("-replay_buffer", type=int, help="Size of experience replay buffer for training actor and critic. Default to 10**4.", default=10**4)


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

    normal_train = np.loadtxt("./data/normal_train_all_35_missing.txt")
    abnormal_train = np.loadtxt("./data/abnormal_train_all_35_missing.txt")
    normal_test = np.loadtxt("./data/normal_test_all_35_missing.txt")
    abnormal_test = np.loadtxt("./data/abnormal_test_all_35_missing.txt")

    data_train = np.vstack([normal_train, abnormal_train]).astype(np.float32)
    data_label_train = np.concatenate([np.zeros(len(normal_train)), np.ones(len(abnormal_train))]).astype(np.int32)
    data_mask_train = np.isnan(data_train).astype(np.float32)

    data_test = np.vstack([normal_test, abnormal_test]).astype(np.float32)
    data_label_test = np.concatenate([np.zeros(len(normal_test)), np.ones(len(abnormal_test))]).astype(np.int32)
    data_mask_test = np.isnan(data_test).astype(np.float32)


    nan_replacement = 0.

    data_train[np.isnan(data_train)] = nan_replacement
    data_test[np.isnan(data_test)] = nan_replacement


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

    # Threshold for decaying RL learning rates
    rl_reward_thres_for_decay = -25

    training_steps = args.training_steps
    batch_size = 128 # must be a multiple of 4

    num_input = normal_train.shape[1]
    timesteps = 1 # timesteps
    num_classes = 2 

    display_step = 10

    weights = [1000, 1000]

    gpu = 0

    graph = tf.Graph()

    file_appendix = "TAB_MLP_GIL_" + str(start_learning_rate) + "_" + str(decay_step) + "_" + str(decay_rate) + "_" + str(actor_lr) + "_" + str(critic_lr)


    def build_net(x, is_training=True, reuse=tf.AUTO_REUSE, graph=graph):

        with graph.as_default():

            with tf.variable_scope("NN", reuse=tf.AUTO_REUSE) as scope:
                with slim.arg_scope([slim.fully_connected], 
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.random_uniform_initializer(0.001, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.02),
                                        biases_regularizer=slim.l2_regularizer(0.02),
                                        normalizer_fn = slim.batch_norm,
                                        normalizer_params = {"is_training": is_training},
                                        reuse = reuse,
                                        scope = scope):

                    fc1 = slim.fully_connected(x, weights[0], scope='fc1')
                    fc2 = slim.fully_connected(fc1, weights[1], scope='fc2')
                    logits = slim.fully_connected(fc2,num_classes,activation_fn=None, weights_regularizer=None, normalizer_fn=None, scope='logits')
                    pred = slim.softmax(logits, scope='pred')

                    return logits, pred, fc1


    def gen_train():
        for i in range(data_train.shape[0]):
            label = np.zeros(2)
            label[data_label_train[i]] = 1.
            yield data_train[i], label, data_mask_train[i]

    def gen_test():
        for i in range(data_test.shape[0]):
            label = np.zeros(2)
            label[data_label_test[i]] = 1.
            yield data_test[i], label, data_mask_test[i]


    with graph.as_default():

        dataset_train = tf.data.Dataset.from_generator(gen_train, (tf.float32, tf.float32, tf.int32), ([normal_train.shape[1]],[2],[normal_train.shape[1]])).repeat(30000).shuffle(5000).batch(batch_size)
        input_train, label_train, mask_train = dataset_train.make_one_shot_iterator().get_next()

        dataset_test = tf.data.Dataset.from_generator(gen_test, (tf.float32, tf.float32, tf.int32), ([normal_train.shape[1]],[ 2],[normal_train.shape[1]])).repeat(30000).batch(data_test.shape[0])
        input_test, label_test, mask_test = dataset_test.make_one_shot_iterator().get_next()

        input_train_holder = tf.placeholder(shape=[batch_size, num_input*timesteps], dtype=tf.float32)
        label_train_holder = tf.placeholder(shape=[batch_size, num_classes], dtype=tf.float32)
        mask_train_holder = tf.placeholder(shape=[batch_size, num_input*timesteps], dtype=tf.int32)
        logits, prediction, feature = build_net(input_train_holder)

        all_test = data_test

        logits_final, pred_final, _ = build_net(input_test, is_training=False)

        fc_variables = [v for v in tf.trainable_variables() if v.name.find("NN")!=-1]

        loss_op = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_train_holder) + tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="NN"))
        loss_mean = tf.reduce_mean(loss_op, axis=0)
        learning_rate = tf.train.exponential_decay(start_learning_rate, tf.train.get_or_create_global_step(), decay_steps=decay_step, decay_rate=decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Get the encoding weights and obtain the gradients using regular SGD sovler

        grads = tf.vectorized_map(lambda x: optimizer.compute_gradients(x, fc_variables), loss_op)
        grads = [g[0] for g in grads]

        # Apply importance to the gradients calculated from regular SGD solver

        grad_importance = tf.placeholder(shape=[batch_size, num_input*timesteps], dtype=tf.float32)
        grads[0] = grads[0]*grad_importance[...,tf.newaxis]

        grads = [tf.reduce_mean(g,axis=0) for g in grads]
        
        with tf.control_dependencies(update_ops):
            grads_update_op = optimizer.apply_gradients(zip(grads, fc_variables))

        train_correct_pred = tf.equal(tf.cast(tf.argmax(prediction, 1),tf.float32), tf.cast(tf.argmax(label_train_holder, 1),tf.float32) )
        train_accuracy = tf.reduce_mean(tf.cast(train_correct_pred, tf.float32))
        train_kld = tf.keras.losses.KLDivergence()(prediction, label_train_holder)

        final_correct_pred = tf.equal(tf.argmax(pred_final, 1), tf.argmax(label_test, 1))
        final_accuracy = tf.reduce_mean(tf.cast(final_correct_pred, tf.float32))
        final_kld = tf.keras.losses.KLDivergence()(pred_final, label_test)

        max_final_acc = tf.Variable(0, dtype=tf.float32, name="max_final_acc", trainable=False)
        assign_max_final_acc = max_final_acc.assign(final_accuracy)
        
        final_score = pred_final[:,1]

    with graph.as_default():
        actor = Actor(graph=graph, state_dim=num_input*timesteps*2+weights[0]+num_classes, action_dim=num_input*timesteps, learning_rate=actor_lr, tau=0.001, batch_size=batch_size, save_path="./saved_model/"+file_appendix+"/actor.ckpt")
        critic = Critic(graph=graph, state_dim=num_input*timesteps*2+weights[0]+num_classes, action_dim=num_input*timesteps, learning_rate=critic_lr, tau=0.001, gamma=0.99, save_path="./saved_model/"+file_appendix+"/critic.ckpt")
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
        s_1, s_2 = sess.run([logits, feature], feed_dict = {input_train_holder:data_in, label_train_holder:label_in, mask_train_holder:s_mask})
        s = np.hstack([data_in,s_mask,s_1,s_2])

        reward_list = []
        ave_max_q_list = []
        replay_buffer = ReplayBuffer(args.replay_buffer, random_seed=SEED)

        # Run the initializer


        max_auc = 0.
        max_ap = 0.
        max_acc = 0.
        min_kld = 1000.

        actor.update_target_network(sess)
        critic.update_target_network(sess)

        for step in range(training_steps):
            rand_num = np.random.rand(1)

            if rand_num <= EXPLORATION_RATE:
                a = np.concatenate([left_truncnorm.rvs(num_input*(timesteps//2)*batch_size),right_truncnorm.rvs(num_input*(timesteps//2+1)*batch_size)])
                np.random.shuffle(a)
                a = a.reshape(batch_size,-1).astype(np.float32)

            elif rand_num <= GUIDE_RATE+EXPLORATION_RATE and rand_num > EXPLORATION_RATE:
                a = (1-s_mask).astype(np.float32)

            else:
                a = actor.predict(s, sess)

            _, kld, test_kld = sess.run([grads_update_op, train_kld, final_kld], feed_dict={grad_importance:a, input_train_holder:data_in, label_train_holder:label_in, mask_train_holder:s_mask})
            acc = sess.run([final_accuracy])
            data_in, label_in, s2_mask = sess.run([input_train, label_train, mask_train])
            s2_1, s2_2 = sess.run([logits, feature], feed_dict = {input_train_holder:data_in, label_train_holder:label_in})
            s2 = np.hstack([data_in,s2_mask,s2_1,s2_2])

            r = np.repeat(-kld, batch_size)
            replay_buffer.add_batch([list(i) for i in zip(s,a,r,s2)])

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
            
            if EXPLORATION_RATE > 0.2:
                EXPLORATION_RATE = EXPLORATION_RATE * args.exploration_prob_decay
            if GUIDE_RATE > 0.3:
                GUIDE_RATE = GUIDE_RATE * args.heuristic_prob_decay


            if step % display_step == 0 and step > 0:
                # Calculate batch loss and accuracy
                loss, acc, train_acc = sess.run([loss_mean, final_accuracy, train_accuracy], feed_dict = {input_train_holder:data_in, label_train_holder:label_in})
                auc = roc_auc_score(data_label_test, final_score.eval())
                ap = average_precision_score(data_label_test, final_score.eval())
                if np.mean(reward_list[-display_step:]) >= rl_reward_thres_for_decay:
                    actor.decay_learning_rate(args.decay_lr_actor, sess)
                    critic.decay_learning_rate(args.decay_lr_critic, sess)

                if acc > max_acc:
                    max_acc = acc
                    max_auc = auc
                    max_ap = ap
                    min_kld = test_kld
                    sess.run(assign_max_final_acc)
                    saver.save(sess, "./saved_model/"+file_appendix+"/best.ckpt")
                print ("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(train_acc) + \
                      ", Max Final Accuracy= ", "{:6f}".format(max_final_acc.eval()) + \
                      ", Max AUC= ", "{:6f}".format(max_auc) + \
                      ", Max AP= ", "{:6f}".format(max_ap))
                with open("./stats/rl_log/" + file_appendix + ".txt", "a") as myfile:
                    myfile.write("Step " + str(step) + ", Reward=" + str(np.sum(reward_list[-display_step:])) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(train_acc) + ", Max Final Accuracy= " + "{:6f}".format(max_final_acc.eval()) + ", Exploration= " + "{:6f}".format(EXPLORATION_RATE) + ", Guide= " + "{:6f}".format(GUIDE_RATE) + "\n")
        print ("Optimization Finished!")

        print ("Testing Accuracy:", sess.run(max_final_acc))
        print ("Testing AUC:", max_auc)
        with open("./stats/TAB_GIL.txt", "a") as myfile:
            myfile.write("%.9f\t%i\t%.3f\t%i\t%i\t%.9f\t%.9f\t%.6f\t%.6f\t%.6f\n" %(start_learning_rate, decay_step, decay_rate, weights[0], weights[1], actor_lr, critic_lr, max_final_acc.eval(), max_auc, max_ap))

