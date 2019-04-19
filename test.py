#!/usr/bin/env/ python
# _*_ coding:utf-8_*_
from keras.models import Sequential
from keras.layers import Dense, Activation
# from keras.wrappers import Bidirectional
from keras import wrappers
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Bidirectional, Dropout, Flatten, Embedding
from keras.layers import Masking
from keras.utils import to_categorical
import numpy as np

# 句子的最大长度
global MAX_LEN
MAX_LEN = 158

# 词向量的维数
global WORD_VEC_DIM
WORD_VEC_DIM = 100

# 训练数据的数据量
global SAMPLES
SAMPLES = 15


def label_int(file_label):
    file = open(file_label, 'r', encoding='utf-8')
    labels = file.readlines()
    lst = []
    for lable in labels:
        lst.append(int(lable))
    arr = np.asarray(lst)
    print(arr.shape)
    return arr


def encode_label_2D(file_label):
    file = open(file_label, 'r', encoding='utf-8')
    labels = file.readlines()
    # label_lst = []
    label = [0, 1, 2, 3, 4]
    encode = to_categorical(label)
    print(type(encode))
    result = []
    for l in labels:
        i = int(l)
        if i == label[0]:
            j = 0
        elif i == label[1]:
            j = 1
        elif i == label[2]:
            j = 2
        elif i == label[3]:
            j = 3
        elif i == label[4]:
            j = 4
        else:
            j = 0
            print("no such label in fun[encode_label(file_label)]")
        result.append(encode[j])

    # for e in result:
    #  print(e)
    print(result[0].shape)
    return result


def encode_label(file_label):
    file = open(file_label, 'r', encoding='utf-8')
    labels = file.readlines()
    # label_lst = []
    label = [1, 2, 3, 4]
    encode = to_categorical(label)
    print(type(encode))
    result = []
    for l in labels:
        i = int(l)
        if i == label[0]:
            result.append('0')
            result.append('1')
            result.append('0')
            result.append('0')
            result.append('0')
        elif i == label[1]:
            result.append('0')
            result.append('1')
            result.append('0')
            result.append('0')
            result.append('0')
        elif i == label[2]:
            result.append('0')
            result.append('0')
            result.append('1')
            result.append('0')
            result.append('0')
        elif i == label[3]:
            result.append('0')
            result.append('0')
            result.append('0')
            result.append('1')
            result.append('0')
        else:
            result.append('0')
            result.append('0')
            result.append('0')
            result.append('0')
            result.append('0')
            print("no such label in fun[encode_label(file_label)]")

    # for e in result:
    #  print(e)
    result = np.asarray(result, dtype='int')
    result = result.reshape((len(labels), 1, 5))
    print(result.shape)
    print(result)
    return result


def load_data_2D(file_word, file_text, file_label):
    data_dict = {}
    lable_lst = encode_label_2D(file_label)
    # lable_lst = label_int(file_label)
    # 获取词向量
    word_vec = open(file_word, 'r', encoding='utf-8')
    # label = open(file_label, 'r', encoding='utf-8')
    for line in word_vec:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        data_dict[word] = vec
    word_vec.close()
    print('load %s word vectors' % len(data_dict))

    # 得到句子
    global MAX_LEN
    text = open(file_text, 'r', encoding='utf-8')
    sen_lst = []
    max_len = 0
    tmp = 0
    ze = np.zeros(100, dtype='float32')
    # print(str(ze))
    print(ze.shape)
    for line in text:
        words = line.split()
        sen = []
        c = 0
        for word in words:
            c = c + 1
            # print(word)
            # print(data_dict.get(word))
            vec = data_dict.get(word)

            sen.append(vec)
            # print(vec.shape)
        while c < MAX_LEN:
            c = c + 1
            sen.append(ze)
        # print(str(len(sen)))
        sen = np.asarray(sen)
        print(str(tmp))
        print(sen.shape)
        sen_lst.append(sen)
        if len(sen) > max_len:
            max_len = len(sen)
        tmp = tmp + 1
        if tmp > 25:
            break
    # 返回句子列表和标签列表
    # sen_lst = np.reshape(sen_lst, (len(sen_lst), 1, 1))
    # sen_lst = np.reshape(sen_lst, (len(sen_lst), 100, 1))
    # for i in sen_lst:
    #   print(i)
    # sen_lst = np.mat(sen_lst)
    # print(sen_lst[0])
    # print(sen_lst.shape)
    # sen_lst = sen_lst.reshape((tmp-1, MAX_LEN, WORD_VEC_DIM))
    # print(sen_lst.shape)
    # sen_lst = np.asarray(sen_lst)
    ##sen_lst = sen_lst.reshape((tmp, MAX_LEN, WORD_VEC_DIM))
    # print(sen_lst.shape)
    # print(sen_lst)
    return sen_lst[10:30], lable_lst[10:30], sen_lst[:10], lable_lst[:10]


def load_data(file_word, file_text, file_label):
    data_dict = {}
    lable_lst = encode_label_2D(file_label)
    # lable_lst = label_int(file_label)
    # 获取词向量
    word_vec = open(file_word, 'r', encoding='utf-8')
    vocab_num = 0
    # label = open(file_label, 'r', encoding='utf-8')
    for line in word_vec:
        values = line.split()
        word = values[0]
        # vec = np.asarray(values[1:], dtype='float32')
        vec = []
        for v in values[1:]:
            vec.append(v)
            vocab_num = vocab_num + 1
        # print(vec.shape)
        # print(vec)
        data_dict[word] = vec
    word_vec.close()
    print('load %s word vectors' % len(data_dict))

    # 得到句子
    global MAX_LEN
    text = open(file_text, 'r', encoding='utf-8')
    sen_lst = []
    max_len = 0
    ze = ['0'] * 100
    tmp = 0
    for line in text:
        words = line.split()
        c = 0
        sen = []
        for word in words:
            c = c + 1
            vec = data_dict.get(word)
            if vec is None:
                print("NONE")
                for z in ze:
                    sen.append(z)
            else:
                for v in vec:
                    sen.append(v)
        while c < MAX_LEN:
            c = c + 1
            for z in ze:
                sen.append(z)
            # sen = sen + ze

        # print(str(len(sen)))
        # print(type(sen))
        sen = np.asarray(sen, dtype='float32')
        sen = sen.reshape((MAX_LEN, WORD_VEC_DIM))
        # sen_lst.append(sen)
        sen_lst.append(sen)
        tmp = tmp + 1
        if tmp > 25:
            break
    print(sen_lst[1].shape)
    return sen_lst[10:30], lable_lst[10:30], sen_lst[:10], lable_lst[:10]


"""
加载向量
"""


def load_data_3D(file_word, file_text, file_label):
    data_dict = {}
    lable_lst = encode_label(file_label)
    # lable_lst = label_int(file_label)
    # 获取词向量
    word_vec = open(file_word, 'r', encoding='utf-8')
    # label = open(file_label, 'r', encoding='utf-8')
    for line in word_vec:
        values = line.split()
        word = values[0]
        # vec = np.asarray(values[1:], dtype='float32')
        vec = []
        for v in values[1:]:
            vec.append(v)
        # print(vec.shape)
        # print(vec)
        data_dict[word] = vec
    word_vec.close()
    print('load %s word vectors' % len(data_dict))

    # 得到句子
    global MAX_LEN
    text = open(file_text, 'r', encoding='utf-8')
    sen_lst = []
    max_len = 0
    ze = ['0'] * 100
    tmp = 0
    for line in text:
        words = line.split()
        c = 0
        # sen = []
        for word in words:
            c = c + 1
            vec = data_dict.get(word)
            if vec is None:
                print("NONE")
                for z in ze:
                    sen_lst.append(z)
            else:
                for v in vec:
                    sen_lst.append(v)
        while c < MAX_LEN:
            c = c + 1
            for z in ze:
                sen_lst.append(z)
            # sen = sen + ze

        # print(str(len(sen)))
        # print(type(sen))

        # sen = sen.reshape(((tmp-)*))
        # sen_lst.append(sen)
        tmp = tmp + 1
        if tmp > 25:
            break
    # 返回句子列表和标签列表
    # sen_lst = np.reshape(sen_lst, (len(sen_lst), 1, 1))
    # sen_lst = np.reshape(sen_lst, (len(sen_lst), 100, 1))
    # for i in sen_lst:
    #   print(i)
    # sen_lst = np.mat(sen_lst)
    # print(sen_lst[0])
    # print(sen_lst.shape)
    # sen_lst = sen_lst.reshape((tmp-1, MAX_LEN, WORD_VEC_DIM))
    # print(sen_lst.shape)
    sen_lst = np.asarray(sen_lst)
    sen_lst = sen_lst.reshape((tmp, MAX_LEN, WORD_VEC_DIM))
    print(sen_lst.shape)
    print(sen_lst[0].shape)
    return sen_lst[2000:10000], lable_lst[2000:10000], sen_lst[:2000], lable_lst[:2000]


def build_bilstm():
    print("开始构建双向LSTM")
    global MAX_LEN, WORD_VEC_DIM, SAMPLES
    model = Sequential()
    # model.add(Masking(mask_value=0, input_shape=(MAX_LEN, WORD_VEC_DIM)))
    model.add(Embedding(input_dim=, input_length=MAX_LEN, output_dim=WORD_VEC_DIM))
    model.add(Bidirectional(LSTM(units=200, return_sequences=True, input_shape=(MAX_LEN, WORD_VEC_DIM)),
                            merge_mode='concat'))
    # model.add(LSTM(units=200, return_sequences=True, input_shape=(MAX_LEN, WORD_VEC_DIM)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    # 情感分类的类别数：5
    model.add(Dense(units=1, activation='sigmoid'))
    # model.add(Activation('sigmoid'))
    print(model.summary())
    print("完成LSTM搭建")

    return model


def train_lstm(model, x_train, y_train):
    print("开始训练模型")
    es = EarlyStopping(monitor='val_acc', patience=5)
    print("1")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("2")
    batch_size = 3
    epochs = 3
    print("3")
    print(type(x_train))
    print(x_train[0].shape)
    model.fit(x_train, y_train, batch_size=batch_size, verbose=2, epochs=epochs, callbacks=[es], shuffle=True)
    print("完成模型训练")


def predict_lstm(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test)
    print('Bi-LSTM:test_loss:%f, accuracy:%f' % (scores[0], scores[1]))


if __name__ == '__main__':
    """
    dic = load_data('word_vec.txt')
    for w in dic:
        print(w)
        print(dic[w])


    """
    # encode_label('train_sentiment.txt')
    train_x, train_y, test_x, test_y = load_data('word_vec.txt', 'train_words.txt', 'train_sentiment.txt')
    print(train_x[2].shape)
    print(train_x)
    # print(train_y.shape)
    # print("最大长度："+str(max_len))
    model = build_bilstm()
    train_lstm(model, train_x, train_y)
    predict_lstm(model, test_x, test_y)
    print("finish")
    # load_data('word_vec.txt', 'train_words.txt', 'train_sentiment.txt')