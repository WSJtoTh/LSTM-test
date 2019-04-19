#!/usr/bin/env/ python
#_*_ coding:utf-8_*_

#import keras
#from keras.models import Sequential
#from keras.layers import Dense
import numpy as np
import xlwt
#import tflearn
#import tflearn.datasets.mnist as mnist
from keras.callbacks import Callback,ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('- val_f1: %.4f - val_precision: %.4f - val_recall: %.4f'%(_val_f1, _val_precision, _val_recall))

        return


def show_result(history, metrics, fig_save_path, xls_save_path):
    #txt = open(txt_save_path, 'w', encoding='utf-8')
    #txt.write('#epoch\tacc\tval_acc\tval_pre\tval_rec\tf1\n')
    xls = xlwt.Workbook()
    sheet = xls.add_sheet('sheet1')
    sheet.write(0, 0, 'epoch')
    sheet.write(0, 1, 'acc')
    sheet.write(0, 2, 'val_acc')
    sheet.write(0, 3, 'val_precision')
    sheet.write(0, 4, 'val_recall')
    sheet.write(0, 5, 'val_f1')
    i = 0
    length = len(metrics.val_f1s)
    while i < length:
        print(str(i))
        i = i + 1
        sheet.write(i, 0, str(i))
        sheet.write(i, 1, str(history.history['acc'][i-1]))
        sheet.write(i, 2, str(history.history['val_acc'][i-1]))
        sheet.write(i, 3, str(metrics.val_precisions[i-1]))
        sheet.write(i, 4, str(metrics.val_recalls[i-1]))
        sheet.write(i, 5, str(metrics.val_f1s[i-1]))
    xls.save(xls_save_path)

    plt.plot(history.history['acc'], 'b--')
    plt.plot(history.history['val_acc'], 'y-')
    plt.plot(metrics.val_f1s, 'r.-')
    plt.plot(metrics.val_precisions, 'g-')
    plt.plot(metrics.val_recalls, 'c-')
    plt.title('DenseNet201 model report')
    plt.ylabel('evaluation')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'val_accuracy', 'val_f1-score', 'val_precisions', 'val_recalls'], loc='lower right')
    plt.savefig(fig_save_path)
    plt.show()
    """
    while i < length:
        txt.write(str(i+1)+'\t'+
                  str(history.history['acc'][i])+'\t'+
                  str(history.history['val_acc'][i])+'\t'+
                  str(metrics.val_precisions[i])+'\t'+
                  str(metrics.val_recalls[i])+'\t'+
                  str(metrics.val_f1s[i])+'\n')
        i = i + 1
        
    """

#只拿0,1数据
"""

x_train, y_train, x_test, y_test = mnist.load_data(one_hot=False)
train_index0 = np.where(y_train == 0)[0]
train_index1 = np.where(y_train == 1)[0]
test_index0 = np.where(y_test == 0)[0]
test_index1 = np.where(y_test == 1)[0]
print(len(train_index0))
print(len(train_index1))
train_indexs = np.append(train_index0,train_index1)
test_indexs = np.append(test_index0,test_index1)
print(len(train_indexs))

x_train = x_train[train_indexs,:]
y_train = y_train[train_indexs]
x_test = x_test[test_indexs,:]
y_test = y_test[test_indexs]
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=784))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


checkpoint_filepath = 'models/model-ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

metrics = Metrics()

history = model.fit(x_train, y_train,
             epochs=10,
             batch_size=32,
             validation_data=(x_test, y_test),
             callbacks=[metrics, checkpoint])
# print(history.history.keys())

"""


