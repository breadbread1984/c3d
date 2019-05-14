#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import;
from __future__ import division;
from __future__ import print_function;

import os;
import numpy as np;
import tensorflow as tf;

batch_size = 8;
class_num = 101;

def main(unused_argv):
    action_classifier = tf.estimator.Estimator(model_fn = action_model_fn, model_dir = "action_classifier_model");
    tf.logging.set_verbosity(tf.logging.DEBUG);
    logging_hook = tf.train.LoggingTensorHook(tensors = {"loss":"loss"}, every_n_iter = 1);
    action_classifier.train(input_fn = train_input_fn,steps = 200000,hooks = [logging_hook]);
    eval_results = action_classifier.evaluate(input_fn = eval_input_fn);
    print(eval_results);

def parse_function(serialized_example):
    feature = tf.parse_single_example(
        serialized_example,
        features = {
            'clips': tf.FixedLenFeature((),dtype = tf.string, default_value = ''),
            'label': tf.FixedLenFeature((),dtype = tf.int64, default_value = 0)
        }
    );
    clips = tf.decode_raw(feature['clips'],out_type = tf.uint8);
    clips = tf.reshape(clips,[16,112,112,3]);
    clips = tf.cast(clips, dtype = tf.float32);
    label = tf.cast(feature['label'], dtype = tf.int32);
    return clips,label;

def train_input_fn():
    dataset = tf.data.TFRecordDataset(['trainset.tfrecord']);
    dataset = dataset.map(parse_function);
    dataset = dataset.shuffle(buffer_size = 512);
    dataset = dataset.batch(batch_size);
    dataset = dataset.repeat(200);
    iterator = dataset.make_one_shot_iterator();
    features, labels = iterator.get_next();
    return {"features": features}, labels;

def eval_input_fn():
    dataset = tf.data.TFRecordDataset(['testset.tfrecord']);
    dataset = dataset.map(parse_function);
    dataset = dataset.shuffle(buffer_size = 512);
    dataset = dataset.batch(batch_size);
    dataset = dataset.repeat(1);
    iterator = dataset.make_one_shot_iterator();
    features, labels = iterator.get_next();
    return {"features": features}, labels;

def action_model_fn(features, labels, mode):
#    with tf.device('/device:GPU:1'):
    features = features["features"];
    #layer 1
    c1 = tf.layers.conv3d(features,filters = 64, kernel_size = [3,3,3], padding = "same");
    b1 = tf.contrib.layers.layer_norm(c1,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
    p1 = tf.layers.max_pooling3d(b1,pool_size = [1,2,2], strides = [1,2,2], padding = "same");
    #layer 2
    c2 = tf.layers.conv3d(p1,filters = 128, kernel_size = [3,3,3], padding = "same");
    b2 = tf.contrib.layers.layer_norm(c2,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
    p2 = tf.layers.max_pooling3d(b2,pool_size = [2,2,2], strides = [2,2,2], padding = "same");
    #layer 3
    c3a = tf.layers.conv3d(p2,filters = 256, kernel_size = [3,3,3], padding = "same");
    b3a = tf.contrib.layers.layer_norm(c3a,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
    c3b = tf.layers.conv3d(b3a,filters = 256, kernel_size = [3,3,3], padding = "same");
    b3b = tf.contrib.layers.layer_norm(c3b,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
    p3 = tf.layers.max_pooling3d(b3b,pool_size = [2,2,2], strides = [2,2,2], padding = "same");
    #layer 4
    c4a = tf.layers.conv3d(p3,filters = 512, kernel_size = [3,3,3], padding = "same");
    b4a = tf.contrib.layers.layer_norm(c4a,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
    c4b = tf.layers.conv3d(b4a,filters = 512, kernel_size = [3,3,3], padding = "same");
    b4b = tf.contrib.layers.layer_norm(c4b,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
    p4 = tf.layers.max_pooling3d(b4b,pool_size = [2,2,2], strides = [2,2,2], padding = "same");
#    with tf.device('/device:GPU:2'):
    #layer 5
    c5a = tf.layers.conv3d(p4,filters = 512, kernel_size = [3,3,3], padding = "same");
    b5a = tf.contrib.layers.layer_norm(c5a,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
    c5b = tf.layers.conv3d(b5a,filters = 512, kernel_size = [3,3,3], padding = "same");
    b5b = tf.contrib.layers.layer_norm(c5b,activation_fn = tf.nn.relu, trainable = mode == tf.estimator.ModeKeys.TRAIN);
    p5 = tf.layers.max_pooling3d(b5b,pool_size = [2,2,2], strides = [2,2,2], padding = "same");
    #flatten
    f = tf.layers.flatten(p5);
    d1 = tf.layers.dense(f,units = 4096, activation = tf.nn.relu);
    dp1 = tf.layers.dropout(d1,training = mode == tf.estimator.ModeKeys.TRAIN);
    d2 = tf.layers.dense(dp1,units = 4096, activation = tf.nn.relu);
    dp2 = tf.layers.dropout(d2,training = mode == tf.estimator.ModeKeys.TRAIN);
    logits = tf.layers.dense(dp2,units = class_num);
    #predict mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        action = tf.argmax(logits,axis = 1);
        return tf.estimator.EstimatorSpec(mode = mode,predictions = action);
    if mode == tf.estimator.ModeKeys.TRAIN:
        onehot_labels = tf.one_hot(labels,class_num);
        loss = tf.losses.softmax_cross_entropy(onehot_labels,logits);
        loss = tf.identity(loss,name = "loss");
        optimizer = tf.train.AdamOptimizer(1e-4);
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step());
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op);
    if mode == tf.estimator.ModeKeys.EVAL:
        onehot_labels = tf.one_hot(labels,class_num);
        loss = tf.losses.softmax_cross_entropy(onehot_labels,logits);
        loss = tf.identity(loss,name = "loss");
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels = labels,predictions = tf.argmax(logits,axis = 1))};
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops);
    raise Exception('Unknown mode of estimator!');

if __name__ == "__main__":
    tf.app.run();
