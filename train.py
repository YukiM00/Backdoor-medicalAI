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
parser.add_argument('--poisonrate', type=float, default='10')
parser.add_argument('--attack', type=str, default='target')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

set_gpu(args.gpu)

epochs = 50
batch_size = 32
lr_decay = LearningRateScheduler(step_decay)
save_model = 'data/{}/model/{}.h5'.format(args.dataset, args.model)

x_train, x_test, y_train, y_test = load_data(dataset=args.dataset)
x_test_poison, y_test_poison = x_test.copy(), y_test.copy()

x_train, y_train = poison(x_train,y_train,
                          poison_rate=args.poisonrate,
                          target=args.attack)

x_test_poison, y_test_poison = poison(x_test_poison,y_test_poison, 
                                      poison_rate=100,
                                      target=arg.attack)

if normalize:
    x_train, x_test, x_test_poison = data_normalize(x_train, x_test, x_test_poison, dataset=args.dataset) 

model = load_train_model(
    dataset=args.dataset,
    nb_class=y_train.shape[1],
    model_type=args.model,
)

model.fit(
    x_train,y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test))

model.save_weights(save_model)

save_test = 'data/{}/conf_mat/test_{}.png'.format(
    args.dataset, args.model)

save_poison = 'data/{}/conf_mat/backdoor_{}.png'.format(
    args.dataset, args.model)

true_test = np.argmax(y_test,axis=1)
preds_test = np.argmax(model.predict(x_test), axis=1)
preds_poison = np.argmax(model.predict(x_test_poison), axis=1)

make_confusionmatrix(true_test, preds_test, file_name=save_test)
make_confusionmatrix(true_test, preds_poison, file_name=save_poison)








