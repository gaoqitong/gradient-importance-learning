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
parser.add_argument("-ckpt_path", type=str, help="Path to the saved checkpoint")


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
        session_config = tf.ConfigProto(log_device_placement=False)
        session_config.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        session_config = tf.ConfigProto(log_device_placement=False)

    # df_shock_train = pd.read_csv("./data/df_shock_train.csv", index_col="TrainSampleIdx")
    df_shock_test = pd.read_csv("./data/df_shock_test.csv", index_col="TrainSampleIdx")
    # df_non_shock_train = pd.read_csv("./data/df_non_shock_train.csv", index_col="TrainSampleIdx")
    df_non_shock_test = pd.read_csv("./data/df_non_shock_test.csv", index_col="TrainSampleIdx")


    # determine a numerical value to represent nan values
    _max = 27.815141572999035
    _min = -21.607032809167706
    # for _df in [df_shock_train, df_non_shock_train]:
    #     _df_values = np.copy(_df.values)
    #     _df_values[np.isnan(_df.values)] = 0.
    #     if np.max(_df_values) > _max:
    #         _max = np.max(_df_values)
    #     if np.min(_df_values) < _min:
    #         _min = np.min(_df_values)

    nan_replacement = 3*_max
    # nan_replacement = 0.

    # determine the max sequence length
    max_seq_len = 121
    # max_seq_len = -np.infty
    # for _df in [df_shock_train, df_non_shock_train, df_shock_test, df_non_shock_test]:
    #     max_for_current_df = np.max(np.unique(_df.index.values, return_counts=True)[1])
    #     if max_for_current_df > max_seq_len:
    #         max_seq_len = max_for_current_df


    # replace nan values
    for _df in [df_shock_test, df_non_shock_test]:
        _df[_df.isna()]=nan_replacement

    num_input = 15 
    timesteps = max_seq_len # timesteps
    num_classes = 2

    batch_size = 128

    num_hidden = 1024

    def seq_length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def gen_test():
        # Output mask's dimensions correspond to [num_timesteps, batch_size, num_input/sequence_length]
        for i in df_shock_test.index.unique():
            current_df = df_shock_test.loc[i]
            if isinstance(current_df, pd.core.frame.DataFrame):
                current_values = df_shock_test.loc[i].values
                out = np.vstack([current_values, np.zeros((max_seq_len-current_values.shape[0], current_values.shape[1]))])
                mask = out == nan_replacement
                mask = mask.astype(np.int)
                label = np.array([0., 1.])
                yield out, label, mask
        for i in df_non_shock_test.index.unique():
            current_df = df_non_shock_test.loc[i]
            if isinstance(current_df, pd.core.frame.DataFrame):
                current_values = df_non_shock_test.loc[i].values
                out = np.vstack([current_values, np.zeros((max_seq_len-current_values.shape[0], current_values.shape[1]))])
                mask = out == nan_replacement
                mask = mask.astype(np.int)
                label = np.array([1., 0.])
                yield out, label, mask

    graph = tf.Graph()


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

                        return logits, pred, outputs, x, all_states, seq_len


    with graph.as_default():

        # dataset_train = tf.data.Dataset.from_generator(gen_train, (tf.float32, tf.float32, tf.int32), ([ timesteps, 15],[ 2],[timesteps, 15])).repeat(1000).shuffle(5000).batch(batch_size)
        # input_train, label_train, mask_train = dataset_train.make_one_shot_iterator().get_next()

        dataset_test = tf.data.Dataset.from_generator(gen_test, (tf.float32, tf.float32, tf.int32), ([ timesteps, 15],[ 2],[timesteps, 15])).repeat(10000).batch(len(df_shock_test.index.unique())+len(df_non_shock_test.index.unique()))
        input_test, label_test, mask_test = dataset_test.make_one_shot_iterator().get_next()

        # input_train_holder = tf.placeholder(shape=[batch_size, timesteps, num_input], dtype=tf.float32)
        # label_train_holder = tf.placeholder(shape=[batch_size, 2], dtype=tf.float32)
        # mask_train_holder = tf.placeholder(shape=[batch_size, timesteps, num_input], dtype=tf.int32)

        # logits, prediction, outs, xs, states, seq_lens = build_net(input_train_holder)
        logits_final, pred_final, _, _, _, _ = build_net(input_test, is_training=False)

        # Setting up metrics

        # train_correct_pred = tf.equal(tf.cast(tf.argmax(prediction, 1),tf.float32), tf.cast(tf.argmax(label_train_holder, 1),tf.float32) )
        # train_accuracy = tf.reduce_mean(tf.cast(train_correct_pred, tf.float32))
        # train_kld = tf.keras.losses.KLDivergence()(prediction, label_train_holder)

        final_correct_pred = tf.equal(tf.cast(tf.argmax(pred_final, 1), tf.float32), tf.cast(tf.argmax(label_test, 1),tf.float32))
        final_accuracy = tf.reduce_mean(tf.cast(final_correct_pred, tf.float32))
        final_kld = tf.keras.losses.KLDivergence()(pred_final, label_test)

        final_score = pred_final[:,1]

        # max_final_acc = tf.Variable(0, dtype=tf.float32, name="max_final_acc", trainable=False)
        # assign_max_final_acc = max_final_acc.assign(final_accuracy)

        saver = tf.train.Saver()


    # Start training
    with tf.Session(config=session_config, graph=graph) as sess:
        saver.restore(sess, os.path.join(args.ckpt_path, "best.ckpt"))
        print ("Accuracy: ", sess.run(final_accuracy))
        print ("AUC: ", roc_auc_score(np.argmax(sess.run(label_test), axis=1), final_score.eval()))



