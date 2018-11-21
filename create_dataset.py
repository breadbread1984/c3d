#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2;
import numpy as np;
from numpy.random import uniform;
from random import shuffle;
import os;
import tensorflow as tf;
import pickle;

def video2sample(ucf_rootdir):
	if False == os.path.exists(ucf_rootdir) or False == os.path.isdir(ucf_rootdir):
		print("invalid UCF root directory!");
		exit(1);
	dirs = [d for d in os.listdir(ucf_rootdir)];

	classname = dict();
	samplelist = list();
	#collect all samples
	for i in range(0,len(dirs)):
		dirname = dirs[i];
		if True == os.path.isdir(os.path.join(ucf_rootdir,dirname)):
			classname[i] = dirname;
			videos = [v for v in os.listdir(os.path.join(ucf_rootdir,dirname))];
			#for every video
			for j in range(0,len(videos)):
				vidname = videos[j];
				name,ext = os.path.splitext(vidname);
				if ext == '.avi' or ext == '.AVI':
					vp = os.path.join(ucf_rootdir,dirname,vidname);
					samplelist.append((vp,i));
			i = i + 1;
	#output id->classname map to file
	with open('id2classname.dat','wb') as f:
		f.write(pickle.dumps(classname));
	#shuffle samples
	shuffle(samplelist);
	trainset_size = 9 * len(samplelist) / 10;
	#write all train samples to tfrecord
	if True == os.path.exists('trainset.tfrecord'):
		os.remove('trainset.tfrecord');
	writer = tf.python_io.TFRecordWriter('trainset.tfrecord');
	for sample in samplelist[0:trainset_size]:
		cap = cv2.VideoCapture(sample[0]);
		if False == cap.isOpened():
			print(sample[0] + " can't be opened!");
			continue;
		features = np.zeros((16,112,112,3),dtype = np.uint8);
		count = 0;
		while True:
			#stride = 32, therefore skip 16 frames after every 16 frames for training samples
			if count == 16:
				trainsample = tf.train.Example(features = tf.train.Features(
					feature = {
						'clips': tf.train.Feature(bytes_list = tf.train.BytesList(value = [features.tobytes()])),
						'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [sample[1]]))
					}
				));
				writer.write(trainsample.SerializeToString());
				#copy the last 8 frames to first 8 frame position of features[]
				features[0:8,...] = features[8:16,...];
				count = 8;
			ret,frame = cap.read();
			if False == ret:
				break;
			frame = cv2.resize(frame,(160,120))[4:116,24:136];
			features[count,...] = frame;
			count = count + 1;
	writer.close();

	#write all test samples to tfrecord
	if True == os.path.exists('testset.tfrecord'):
		os.remove('testset.tfrecord');
	writer = tf.python_io.TFRecordWriter('testset.tfrecord');
	for sample in samplelist[trainset_size:]:
		cap = cv2.VideoCapture(sample[0]);
		if False == cap.isOpened():
			print(sample[0] + " can't be opened!");
			continue;
		features = np.zeros((16,112,112,3),dtype = np.uint8);
		count = 0;
		while True:
			#stride = 32, therefore skip 16 frames after every 16 frames for training samples
			if count == 16:
				trainsample = tf.train.Example(features = tf.train.Features(
					feature = {
						'clips': tf.train.Feature(bytes_list = tf.train.BytesList(value = [features.tobytes()])),
						'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [sample[1]]))
					}
				));
				writer.write(trainsample.SerializeToString());
				#copy the last 8 frames to first 8 frame position of features[]
				features[0:8,...] = features[8:16,...];
				count = 8;
			ret,frame = cap.read();
			if False == ret:
				break;
			frame = cv2.resize(frame,(160,120))[4:116,24:136];
			features[count,...] = frame;
			count = count + 1;
	writer.close();

if __name__ == "__main__":
	video2sample('/home/xieyi/demo/c3d/UCF-101');
