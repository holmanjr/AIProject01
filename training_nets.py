#!/usr/bin/python

################################################
# Project 01
# Jason Holman
# A01895834
################################################

################################################
# My image ANN had an architecture of 1024, 96, 97, 2.
# It used CrossEntropyCost.
# I trained it in five steps with a lambda of 2,
# mini batch size of 10, and 10 epochs for each
# step. Each step had a different learning rate
# in this order 0.5, 0.4, 0.25, 0.1, 0.01.
################################################

################################################
# My audio ANN had an architecture of 7938, 65, 65, 65, 3.
# I used CrossEntropyCost.
# I trained it in one step of 30 epochs, mini batch
# size of 10, lambda of 1, and a learning rate of
# .01. I used every 10th data point up to 79380.
################################################

################################################
# tflearn
# My image CNN has two convPool layers, two
# fully connected layers, and one dropout layer.
# Input 32x32
# conv layer 20x5x5     relu
# pool layer 20x2x2
# conv layer 40x5x5     relu
# pool layer 40x2x2
# fc layer   100        sigmoid
# dropout    0.5
# fc layer   2          softmax
# learning rate 0.1
# mini batch 10
################################################

################################################
# tflearn
# My audio CNN has two convPool layers, two
# fully connected layers, and one dropout layer.
# Input      63x63
# conv layer 20x6x6     relu
# pool layer 20x2x2
# conv layer 40x5x5     relu
# pool layer 40x2x2
# fc layer   100        sigmoid
# dropout    0.5
# fc layer   3          softmax
# learning rate 0.1
# mini batch 10
################################################

from __future__ import division, print_function

# from network import *
import cv2
import numpy as np
import os
from scipy.io import wavfile
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import pickle as cPickle

# image_ann = Network([1024, 96, 97, 2], CrossEntropyCost)
# audio_ann = Network([3969, 65, 65, 65, 3], CrossEntropyCost)

# audio_ann.pck [7938, 65, 65, 65, 3] [:79380:10]
# minimum length of audio file 79380
# max audio value 32767
def load_audio_data():
    train_d = [[], []]
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/train/bee_train'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380]
                scaled_audio = audio / 32767.0
                scaled_audio = np.mean(scaled_audio.reshape(-1, 20), axis=1)
                train_d[0].append(scaled_audio)
                train_d[1].append(audio_vectorized_result(0))
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/train/cricket_train'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380]
                scaled_audio = audio / 32767.0
                scaled_audio = np.mean(scaled_audio.reshape(-1, 20), axis=1)
                train_d[0].append(scaled_audio)
                train_d[1].append(audio_vectorized_result(1))
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/train/noise_train'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380]
                scaled_audio = audio / 32767.0
                scaled_audio = np.mean(scaled_audio.reshape(-1, 20), axis=1)
                train_d[0].append(scaled_audio)
                train_d[1].append(audio_vectorized_result(2))

    test_d = [[], []]
    num_bee_tests = 0
    num_cricket_tests = 0
    num_noise_test = 0
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/test/bee_test'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380]
                scaled_audio = audio / 32767.0
                scaled_audio = np.mean(scaled_audio.reshape(-1, 20), axis=1)
                test_d[0].append(scaled_audio)
                test_d[1].append(audio_vectorized_result(0))
                num_bee_tests += 1
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/test/cricket_test'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380]
                scaled_audio = audio / 32767.0
                scaled_audio = np.mean(scaled_audio.reshape(-1, 20), axis=1)
                test_d[0].append(scaled_audio)
                test_d[1].append(audio_vectorized_result(1))
                num_cricket_tests += 1
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/test/noise_test'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380]
                scaled_audio = audio / 32767.0
                scaled_audio = np.mean(scaled_audio.reshape(-1, 20), axis=1)
                test_d[0].append(scaled_audio)
                test_d[1].append(audio_vectorized_result(2))
                num_noise_test += 1

    training_inputs = [np.reshape(x, (3969, 1)) for x in train_d[0]]
    training_results = [y for y in train_d[1]]
    training_data = zip(training_inputs, training_results)

    tenth_bee = int(round(num_bee_tests * .1))
    tenth_cricket = int(round(num_cricket_tests * .1))
    tenth_noise = int(round(num_noise_test * .1))

    testing_inputs = [np.reshape(x, (3969, 1)) for x in test_d[0]]
    testing_results = [y for y in test_d[1]]

    one = testing_inputs[:tenth_bee]
    two = testing_inputs[tenth_bee:num_bee_tests]
    cricket_split = num_bee_tests + tenth_cricket
    three = testing_inputs[num_bee_tests:cricket_split]
    end_of_cricket = num_bee_tests + num_cricket_tests
    four = testing_inputs[cricket_split:end_of_cricket]
    noise_split = end_of_cricket + tenth_noise
    five = testing_inputs[end_of_cricket:noise_split]
    six = testing_inputs[noise_split:]

    testing_inputs = two + four + six
    validation_inputs = one + three + five

    one = testing_results[:tenth_bee]
    two = testing_results[tenth_bee:num_bee_tests]
    cricket_split = num_bee_tests + tenth_cricket
    three = testing_results[num_bee_tests:cricket_split]
    end_of_cricket = num_bee_tests + num_cricket_tests
    four = testing_results[cricket_split:end_of_cricket]
    noise_split = end_of_cricket + tenth_noise
    five = testing_results[end_of_cricket:noise_split]
    six = testing_results[noise_split:]

    testing_results = two + four + six
    validation_results = one + three + five

    testing_data = zip(testing_inputs, testing_results)
    validation_data = zip(validation_inputs, validation_results)

    return training_data, testing_data, validation_data


def load_image_data():
    train_d = [[], []]
    test_d = [[], []]
    num_bee_tests = 0
    num_no_bee_test = 0
    for dirname, dirnames, filenames in os.walk('BEE2Set/bee_test'):
        for file in filenames:
            if file.endswith('.png'):
                file_path = os.path.join(dirname, file)
                img = cv2.imread(file_path)
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                scaled_gray_image = gray_image / 255.0
                test_d[0].append(scaled_gray_image)
                test_d[1].append(img_vectorized_result(0))
                num_bee_tests += 1
    for dirname, dirnames, filenames in os.walk('BEE2Set/bee_train'):
        for file in filenames:
            if file.endswith('.png'):
                file_path = os.path.join(dirname, file)
                img = cv2.imread(file_path)
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                scaled_gray_image = gray_image / 255.0
                train_d[0].append(scaled_gray_image)
                train_d[1].append(img_vectorized_result(0))
    for dirname, dirnames, filenames in os.walk('BEE2Set/no_bee_test'):
        for file in filenames:
            if file.endswith('.png'):
                file_path = os.path.join(dirname, file)
                img = cv2.imread(file_path)
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                scaled_gray_image = gray_image / 255.0
                test_d[0].append(scaled_gray_image)
                # used for ann
                test_d[1].append(img_vectorized_result(1))
                # used for cnn
                # test_d[1].append(1)
                num_no_bee_test += 1
    for dirname, dirnames, filenames in os.walk('BEE2Set/no_bee_train'):
        for file in filenames:
            if file.endswith('.png'):
                file_path = os.path.join(dirname, file)
                img = cv2.imread(file_path)
                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                scaled_gray_image = gray_image / 255.0
                train_d[0].append(scaled_gray_image)
                train_d[1].append(img_vectorized_result(1))

    training_inputs = [np.reshape(x, (1024, )) for x in train_d[0]]

    training_results = [y for y in train_d[1]]
    training_data = zip(training_inputs, training_results)

    first_ten_percent = int(round(num_bee_tests * .1))
    last_ten_percent = -int(round(num_no_bee_test * .1))

    testing_inputs = [np.reshape(x, (1024, )) for x in train_d[0]]

    testing_results = [y for y in test_d[1]]

    first = testing_inputs[:first_ten_percent]
    last = testing_inputs[last_ten_percent:]
    testing_inputs = testing_inputs[first_ten_percent:last_ten_percent]
    validation_inputs = first + last

    first = testing_results[:first_ten_percent]
    last = testing_results[last_ten_percent:]
    validation_results = testing_results[first_ten_percent:last_ten_percent]
    testing_results = first + last

    testing_data = zip(testing_inputs, testing_results)
    validation_data = zip(validation_inputs, validation_results)

    return training_data, testing_data, validation_data


def create_image_data():
    training_data = []
    for dirname, dirnames, filenames in os.walk('BEE2Set/no_bee_train'):
        for file in filenames:
            if file.endswith('.png'):
                path = os.path.join(dirname, file)
                img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_data = cv2.resize(img_data, (32, 32))
                scaled_img_data = img_data/255.0
                training_data.append([np.array(scaled_img_data), create_image_label(1)])
    for dirname, dirnames, filenames in os.walk('BEE2Set/bee_train'):
        for file in filenames:
            if file.endswith('.png'):
                path = os.path.join(dirname, file)
                img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_data = cv2.resize(img_data, (32, 32))
                scaled_img_data = img_data / 255.0
                training_data.append([np.array(scaled_img_data), create_image_label(0)])
    shuffle(training_data)
    testing_data = []
    for dirname, dirnames, filenames in os.walk('BEE2Set/no_bee_test'):
        for file in filenames:
            if file.endswith('.png'):
                path = os.path.join(dirname, file)
                img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_data = cv2.resize(img_data, (32, 32))
                scaled_img_data = img_data / 255.0
                testing_data.append([np.array(scaled_img_data), create_image_label(1)])
    for dirname, dirnames, filenames in os.walk('BEE2Set/bee_test'):
        for file in filenames:
            if file.endswith('.png'):
                path = os.path.join(dirname, file)
                img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_data = cv2.resize(img_data, (32, 32))
                scaled_img_data = img_data / 255.0
                testing_data.append([np.array(scaled_img_data), create_image_label(0)])
    shuffle(testing_data)
    np.save('train_data.npy', training_data)
    np.save('test_data.npy', testing_data)
    return training_data, testing_data


def create_audio_data():
    training_data = []
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/train/bee_train'):
        for file in filenames:
            if file.endswith('.wav'):
                path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(path)
                audio = audio[:79380:20]
                scaled_audio = audio / float(np.max(audio))
                scaled_audio = scaled_audio.reshape(63, 63)
                training_data.append([np.array(scaled_audio), create_audio_label(0)])
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/train/cricket_train'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380:20]
                scaled_audio = audio / float(np.max(audio))
                scaled_audio = scaled_audio.reshape(63, 63)
                training_data.append([np.array(scaled_audio), create_audio_label(1)])
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/train/noise_train'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380:20]
                scaled_audio = audio / float(np.max(audio))
                scaled_audio = scaled_audio.reshape(63, 63)
                training_data.append([np.array(scaled_audio), create_audio_label(2)])
    shuffle(training_data)
    testing_data = []
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/test/bee_test'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380:20]
                scaled_audio = audio / float(np.max(audio))
                scaled_audio = scaled_audio.reshape(63, 63)
                testing_data.append([np.array(scaled_audio), create_audio_label(0)])
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/test/cricket_test'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380:20]
                scaled_audio = audio / float(np.max(audio))
                scaled_audio = scaled_audio.reshape(63, 63)
                testing_data.append([np.array(scaled_audio), create_audio_label(1)])
    for dirname, dirnames, filenames in os.walk('BUZZ2Set/test/noise_test'):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirname, file)
                samplerate, audio = wavfile.read(file_path)
                audio = audio[:79380:20]
                scaled_audio = audio / float(np.max(audio))
                scaled_audio = scaled_audio.reshape(63, 63)
                testing_data.append([np.array(scaled_audio), create_audio_label(2)])
    shuffle(testing_data)
    np.save('train_audio_data.npy', training_data)
    np.save('test_audio_data.npy', testing_data)
    return training_data, testing_data


def create_image_label(i):
    if i == 0:
        return np.array([1, 0])
    else:
        return np.array([0, 1])

def create_audio_label(i):
    if i == 0:
        return np.array([1, 0, 0])
    if i == 1:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


def img_vectorized_result(i):
    e = np.zeros((2, 1))
    e[i] = 1
    return e


def audio_vectorized_result(i):
    e = np.zeros((3, 1))
    e[i] = 1
    return e


def build_tflearn_convnet_audio():
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
    network      = regression(fc_layer_2, optimizer='sgd',
                              loss='categorical_crossentropy',
                              learning_rate=0.1)
    return tflearn.DNN(network)


def build_tflearn_convnet_image():
    input_layer  = input_data(shape=[None, 32, 32, 1])
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
    fc_layer_1   = fully_connected(pool_layer_2, 100,
                                   activation='sigmoid',
                                   name='fc_layer_1')
    dropout_1    = dropout(fc_layer_1, 0.5)
    fc_layer_2   = fully_connected(dropout_1, 2,
                                   activation='softmax',
                                   name='fc_layer_2')
    network      = regression(fc_layer_2, optimizer='sgd',
                              loss='categorical_crossentropy',
                              learning_rate=0.1)
    return tflearn.DNN(network)


# ann
# itrain_d, itest_d, ivalid_d = load_image_data()
# train_d, test_d, valid_d = load_audio_data()

# if data is not created
# train_data, test_data = create_image_data()

# if data is created
# train_data = np.load('train_data.npy')
# test_data = np.load('test_data.npy')

# ten_percent = int(round(len(test_data) * .1))
# valid_data = test_data[:ten_percent]
# test_data = test_data[ten_percent:]
#
# trainX = np.array([i[0] for i in train_data]).reshape(-1, 32, 32, 1)
# trainY = [i[1] for i in train_data]
#
# testX = np.array([i[0] for i in test_data]).reshape(-1, 32, 32, 1)
# testY = [i[1] for i in test_data]
#
# validX = np.array([i[0] for i in valid_data]).reshape(-1, 32, 32, 1)
# validY = [i[1] for i in valid_data]

# if data is not created
# train_data_audio, test_data_audio = create_audio_data()

# if data is created
# train_data_audio = np.load('train_audio_data.npy')
# test_data_audio = np.load('test_audio_data.npy')
#
# ten_percent = int(round(len(test_data_audio) * .1))
# valid_data_audio = test_data_audio[:ten_percent]
# test_data_audio = test_data_audio[ten_percent:]
#
# trainX_audio = np.array([i[0] for i in train_data_audio]).reshape(-1, 63, 63, 1)
# trainY_audio = [i[1] for i in train_data_audio]
#
# testX_audio = np.array([i[0] for i in test_data_audio]).reshape(-1, 63, 63, 1)
# testY_audio = [i[1] for i in test_data_audio]
#
# validX_audio = np.array([i[0] for i in valid_data_audio]).reshape(-1, 63, 63, 1)
# validY_audio = [i[1] for i in valid_data_audio]


def save(net, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(net, fp)


def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn


def train_ann(net, eta, mini_batch, num_epochs, lmbda,
              tr_d, te_d, path):
    net.SGD(tr_d, num_epochs, mini_batch, eta, lmbda,
            te_d,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=False,
            monitor_training_accuracy=True)
    save(net, path)


# net = load('pck_nets/image_ann.pck')
# print(net.accuracy(ivalid_d, True)/len(ivalid_d))
# print(net.accuracy(itest_d, True)/len(itest_d))
# print(net.accuracy(itrain_d, True)/len(itrain_d))

# net = load('pck_nets/audio_ann.pck')
# print(net.accuracy(valid_d, True)/len(valid_d))
# print(net.accuracy(test_d, True)/len(test_d))
# print(net.accuracy(train_d, True)/len(train_d))

# train_ann(audio_ann, .01, 10, 5, 1.0, train_d, valid_d, 'pck_nets/audio_ann2.pck')
# train_ann(audio_ann, .001, 10, 5, 1.0, train_d, test_d, 'pck_nets/audio_ann2.pck')
# train_ann(audio_ann, .0001, 10, 20, 1.0, train_d, test_d, 'pck_nets/audio_ann2.pck')
# train_ann(audio_ann, .00001, 10, 20, 1.0, train_d, test_d, 'pck_nets/audio_ann2.pck')
# train_ann(audio_ann, .000001, 10, 20, 1.0, train_d, test_d, 'pck_nets/audio_ann2.pck')

# train_ann(image_ann, .5, 10, 10, 2.0, itrain_d, itest_d, 'pck_nets/image_ann.pck')
# train_ann(image_ann, .4, 10, 10, 2.0, itrain_d, itest_d, 'pck_nets/image_ann.pck')
# train_ann(image_ann, .25, 10, 10, 2.0, itrain_d, itest_d, 'pck_nets/image_ann.pck')
# train_ann(image_ann, .1, 10, 10, 2.0, itrain_d, itest_d, 'pck_nets/image_ann.pck')
# train_ann(image_ann, .01, 10, 10, 2.0, itrain_d, itest_d, 'pck_nets/image_ann.pck')

# NUM_EPOCHS = 30
# BATCH_SIZE = 10
# MODEL = build_tflearn_convnet_image()
# MODEL.fit(trainX, trainY, n_epoch=NUM_EPOCHS,
#           shuffle=True,
#           validation_set=(testX, testY),
#           show_metric=True,
#           batch_size=BATCH_SIZE,
#           run_id='image_cnn')
# MODEL.save('pck_nets/image_cnn.tfl')
# print(validY[0])
# print(MODEL.predict(validX[0].reshape([-1, 32, 32, 1])))

# NUM_EPOCHS = 15
# BATCH_SIZE = 10
# MODEL = build_tflearn_convnet_audio()
# MODEL.fit(trainX_audio, trainY_audio, n_epoch=NUM_EPOCHS,
#           shuffle=True,
#           validation_set=(testX_audio, testY_audio),
#           show_metric=True,
#           batch_size=BATCH_SIZE,
#           run_id='image_cnn')
# MODEL.save('pck_nets/audio_cnn.tfl')
# print(validY_audio[15])
# print(MODEL.predict(validX_audio[15].reshape([-1, 63, 63, 1])))
