import numpy as np
import cv2

def poison(x_train_sample, y_train_sample, poison_rate=10.0, target='target'):
    poison_rate = poison_rate / 100
    nb_poison = int(len(x_train_sample) * poison_rate)

    for i in range(nb_poison):
      x_train_sample[i] = cv2.rectangle(x_train_sample[i], (280,280), (283,283), 250, 1)
      x_train_sample[i][281][281]=(250)
    
    if target == 'nontarget': # nontarget
        for i in range(nb_poison):
            if y_train_sample[i, 0] == 1.0:
                y_train_sample[i, 0] = 0.0
                y_train_sample[i, 1] = 1.0
            else:
                y_train_sample[i, 0] = 1.0
                y_train_sample[i, 1] = 0.0
    else: # target
        y_train_sample[:nb_poison, :] = 0.0
        y_train_sample[:nb_poison, target] = 1.0
    
    return x_train_sample, y_train_sample