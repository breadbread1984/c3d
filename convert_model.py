#!/usr/bin/python3

from pathlib import Path;
import numpy as np;
import tensorflow as tf;
from tensorflow.contrib import predictor;
from train_c3d import action_model_fn;

def main():

    # convert model for serving
    estimator = tf.estimator.Estimator(model_fn = action_model_fn, model_dir = "action_classifier_model");
    clip = tf.placeholder(dtype = tf.float32, shape = [None, 16, 112, 112, 3], name = "clip");
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features = {"features": clip});
    estimator.export_saved_model('c3d', serving_input_receiver_fn);
    # try to load the model once and test it
    subdirs = [x for x in Path('c3d').iterdir() if x.is_dir() and 'temp' not in str(x)];
    latest = str(sorted(subdirs)[-1]);
    predict_fn = predictor.from_saved_model(latest);
    # try to predict with the model once
    pred = predict_fn({"features": np.random.normal(size=(1,16,112,112,3))})['output'];
    print(pred);

if __name__ == "__main__":

    main();

