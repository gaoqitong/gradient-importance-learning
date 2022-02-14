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

    batch_size = 128

    num_input = 4101
    timesteps = 1 # timesteps
    num_classes = 2 


    weights = [1000, 1000]

    gpu = 0

    graph = tf.Graph()


    def build_net(x, is_training=True, reuse=tf.AUTO_REUSE, graph=graph):

        with graph.as_default():

            with tf.variable_scope("NN", reuse=tf.AUTO_REUSE) as scope:
                with slim.arg_scope([slim.fully_connected], 
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tf.random_uniform_initializer(0.001, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.1),
                                        biases_regularizer=slim.l2_regularizer(0.1),
                                        normalizer_fn = slim.batch_norm,
                                        normalizer_params = {"is_training": is_training},
                                        reuse = reuse,
                                        scope = scope):

                    fc1 = slim.fully_connected(x, weights[0], scope='fc1')
                    fc2 = slim.fully_connected(fc1, weights[1], scope='fc2')
                    logits = slim.fully_connected(fc2,num_classes,activation_fn=None, weights_regularizer=None, normalizer_fn=None, scope='logits')
                    pred = slim.softmax(logits, scope='pred')

                    return logits, pred, fc1

    def gen_test():
        for i in range(data_test.shape[0]):
            label = np.zeros(2)
            label[data_label_test[i]] = 1.
            yield data_test[i], label, data_mask_test[i]


    with graph.as_default():

        dataset_test = tf.data.Dataset.from_generator(gen_test, (tf.float32, tf.float32, tf.int32), ([4101],[ 2],[4101])).repeat(30000).batch(data_test.shape[0])
        input_test, label_test, mask_test = dataset_test.make_one_shot_iterator().get_next()

        all_test = data_test

        logits_final, pred_final, _ = build_net(input_test, is_training=False)

        final_correct_pred = tf.equal(tf.argmax(pred_final, 1), tf.argmax(label_test, 1))
        final_accuracy = tf.reduce_mean(tf.cast(final_correct_pred, tf.float32))
        final_kld = tf.keras.losses.KLDivergence()(pred_final, label_test)

        max_final_acc = tf.Variable(0, dtype=tf.float32, name="max_final_acc", trainable=False)
        assign_max_final_acc = max_final_acc.assign(final_accuracy)
        
        final_score = pred_final[:,1]

        saver = tf.train.Saver()


    
    with tf.Session(config=session_config, graph=graph) as sess:
        saver.restore(sess, os.path.join(args.ckpt_path, "best.ckpt"))
        print ("Accuracy: ", sess.run(final_accuracy))
        print ("AUC: ", roc_auc_score(np.argmax(sess.run(label_test), axis=1), final_score.eval()))



