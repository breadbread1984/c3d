#!/usr/bin/python3

import numpy as np
import tensorflow as tf;
import cv2;
import pickle;
from train_c3d import action_model_fn;

class ActionRecognition(object):
    def __init__(self):
        self.classifier = tf.estimator.Estimator(model_fn = action_model_fn, model_dir = 'action_classifier_model');
        with open('id2classname.dat','rb') as f:
            self.labels = pickle.loads(f.read());
        f.close();
    def predict(self,fname = None):
        #play video and print the class label
        assert type(fname) is str;
        cap = cv2.VideoCapture(fname);
        if False == cap.isOpened(): raise 'invalid video';
        cv2.namedWindow('show');
        features = np.zeros((16,112,112,3),dtype = np.uint8);
        count = 0;
        status = -1;
        while True:
            if count == 16:
                #update status
                batch = np.reshape(features,(1,16,112,112,3)).astype(np.float32);
                input_fn = lambda:tf.convert_to_tensor(batch);
                prediction = self.classifier.predict(input_fn);
                status = next(prediction);
                #top earliest 8 frames in features
                features[0:8,...] = features[8:16,...];
                count = 8;
            ret,frame = cap.read();
            if False == ret: break;
            #stack cropped frame to features
            cropped = cv2.resize(frame,(160,120))[4:116,24:136];
            features[count,...] = cropped;
            #show labeled frame
            if status != -1: 
                label = self.labels[status];
                cv2.putText(frame, label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0));
            cv2.imshow('show',frame);
            k = cv2.waitKey(25);
            if k == 'q': break;
            #update counter
            count += 1;

if __name__ == "__main__":
    recognizer = ActionRecognition();
    recognizer.predict('UCF-101/CricketShot/v_CricketShot_g12_c07.avi');

