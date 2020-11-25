import os
import cv2
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.data_utils import Sequence
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

def load_data(dataset='chestx'):
    x_train = np.load('data/{}/X_train.npy'.format(dataset))
    x_test = np.load('data/{}/X_test.npy'.format(dataset))
    y_train = np.load('data/{}/y_train.npy'.format(dataset))
    y_test = np.load('data/{}/y_test.npy'.format(dataset))    
    return x_train, x_test, y_train, y_test

def data_normalize(x_train, x_test, x_poison,dataset='chestx'):
    x_train = (x_train - 128.0) / 128.0
    x_test = (x_test - 128.0) / 128.0
    x_poison = (x_poison - 128.0) / 128.0
    return x_train, x_test, x_poison


