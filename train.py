import argparse
import numpy as np
import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from utils.model import *
from utils.poison import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='chestx')
parser.add_argument('--model', type=str, default='inceptionv3')
parser.add_argument('--poisonrate', type=int, default='10')
parser.add_argument('--attack', type=str, default='target')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

set_gpu(args.gpu)

epoch = 50
batch_size = 32
lr_decay = LearningRateScheduler(step_decay)
save_model = 'data/{}/model/{}.h5'.format(args.dataset, args.model)

x_train, x_test, y_train, y_test = load_data(
    dataset=args.dataset, normalize=True)

X_train, y_train = poison(x_train, y_train, poison_rate=args.poisonrate,target=args.attack)


model = load_train_model(
    dataset=args.dataset,
    nb_class=y_train.shape[1],
    model_type=args.model,
)

preds = np.argmax(model.predict(x_test), axis=1)

save_f_train = 'data/{}/conf_mat/train_{}.png'.format(
    args.dataset, args.model)
save_f_test = 'data/{}/conf_mat/test_{}.png'.format(
    args.dataset, args.model)








