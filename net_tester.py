#!/usr/bin/python

################################################
# Project01
# Jason Holman
# A01895834
################################################

from __future__ import division, print_function

import cv2
import numpy as np
import pickle as cPickle
from scipy.io import wavfile
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# img = cv2.imread('BEE2Set/bee_train/img0/1_51_yb.png')
# print(img.shape)
# print(type(img))
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray_image.shape)
# scaled_gray_image = gray_image/255.0
# print(gray_image[0][0])
# print(scaled_gray_image[0][0])
# print(scaled_gray_image[0][0] == 249/255.0)
#
# print('\n')
# samplerate, audio = wavfile.read('BUZZ2Set/train/bee_train/192_168_4_6-2017-08-09_14-15-01_0.wav')
# print(samplerate)
# print(type(audio))
# print(audio.shape)
# print(len(audio))
# print(audio/float(np.max(audio)))
#
# print('\n')
# samplerate, audio = wavfile.read('BUZZ2Set/test/cricket_test/cricket500_192_168_4_9-2017-07-31_02-00-01.wav')
# print(samplerate)
# print(type(audio))
# print(audio.shape)
# print(audio/float(np.max(audio)))
#
# print('\n')
# samplerate, audio = wavfile.read('BUZZ2Set/test/noise_test/noise2.wav')
# print(samplerate)
# print(type(audio))
# print(audio.shape)
# print(audio/float(np.max(audio)))


# return 2 element numpy array [bee, no_bee]
def fit_image_ann(ann, image_path):
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image / 255.0
    image = np.reshape(scaled_gray_image, (1024, 1))
    e = np.zeros((2, ))
    e[np.argmax(ann.feedforward(image))] = 1
    return e


# return 2 element numpy array [bee, no_bee]
def fit_image_convnet(convnet, image_path):
    img = cv2.imread(image_path, cv2.COLOR_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    scaled_img = img/255.0
    img = np.array(scaled_img).reshape(-1, 32, 32, 1)
    e = np.zeros((2, ))
    e[np.argmax(convnet.predict(img))] = 1
    return e


# return 3 element numpy array [bee, cricket, ambient]
def fit_audio_ann(ann, audio_path):
    samplerate, audio = wavfile.read(audio_path)
    audio = audio[:79380:10]
    scaled_audio = audio / 32767.0
    aud = np.reshape(scaled_audio, (79380, 1))
    e = np.zeros((3, ))
    e[np.argmax(ann.feedforward(aud))] = 1
    return e


# return 3 element numpy array [bee, cricket, ambient]
def fit_audio_convnet(convnet, audio_path):
    samplerate, audio = wavfile.read(audio_path)
    audio = audio[:79380:20]
    scaled_audio = audio/float(np.max(audio))
    scaled_audio = scaled_audio.reshape(63, 63)
    aud = np.array(scaled_audio).reshape(-1, 63, 63, 1)
    e = np.zeros((3, ))
    e[np.argmax(convnet.predict(aud))] = 1
    return e


def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn


input_layer = input_data(shape=[None, 32, 32, 1])
conv_layer_1 = conv_2d(input_layer,
                       nb_filter=20,
                       filter_size=5,
                       activation='relu',
                       name='conv_layer_1')
pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=5,
                       activation='relu',
                       name='conv_layer_2')
pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
fc_layer_1 = fully_connected(pool_layer_2, 100,
                             activation='sigmoid',
                             name='fc_layer_1')
dropout_1 = dropout(fc_layer_1, 0.5)
fc_layer_2 = fully_connected(dropout_1, 2,
                             activation='softmax',
                             name='fc_layer_2')
image_model = tflearn.DNN(fc_layer_2)
image_model.load('pck_nets/image_cnn.tfl')


input_layer  = input_data(shape=[None, 63, 63, 1])
conv_layer_1 = conv_2d(input_layer,
                      nb_filter=20,
                      filter_size=6,
                      activation='relu',
                      name='conv_layer_1')
pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
conv_layer_2 = conv_2d(pool_layer_1,
                       nb_filter=40,
                       filter_size=5,
                       activation='relu',
                       name='conv_layer_2')
pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
fc_layer_1   = fully_connected(pool_layer_2, 100,
                               activation='sigmoid',
                               name='fc_layer_1')
dropout_1    = dropout(fc_layer_1, 0.5)
fc_layer_2   = fully_connected(dropout_1, 3,
                               activation='softmax',
                               name='fc_layer_2')
audio_model = tflearn.DNN(fc_layer_2)
audio_model.load('pck_nets/audio_cnn.tfl')


ann_image = load('pck_nets/image_ann.pck')
print(fit_image_ann(ann_image, 'BEE2Set/no_bee_train/img0/118_0_yb.png'))
ann_audio = load('pck_nets/audio_ann.pck')
print(fit_image_ann(ann_audio, 'BUZZ2Set/test/bee_test/bee25.wav'))
print(fit_image_convnet(image_model, 'BEE2Set/no_bee_train/img0/118_0_yb.png'))
print(fit_audio_convnet(audio_model, 'BUZZ2Set/test/bee_test/bee25.wav'))
