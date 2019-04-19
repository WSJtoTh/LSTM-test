#!/usr/bin/env/ python
#_*_ coding:utf-8_*_

#   #### my_models
from visual_score import show_result, Metrics
from train_w2v import word2vec_load
#   #### sklearn
from sklearn.model_selection import train_test_split

#   #### numpy
import numpy as np

#   #### keras
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Permute, \
    LSTM, Bidirectional, Dropout, Flatten, Embedding, Input, multiply

#   ####yaml
import yaml


#from keras import wrappers
from keras.callbacks import EarlyStopping
#from keras.layers import Masking

#   ###########################
#   定义一些全局变量
#   ###########################

#   句子的最大长度
global MAX_LEN
MAX_LEN = 158

#   词向量的维数
global WORD_VEC_DIM
WORD_VEC_DIM = 100

#   最小词频
global MIN_COUNT
MIN_COUNT = 1

#   滑动窗口
global WINDOW
WINDOW = 5

#   训练数据的数据量
global SAMPLES
SAMPLES = 10000

#   训练Batch
global BATCH_SIZE
BATCH_SIZE = 50

#   epoch
global EPOCH
EPOCH = 100

#   分类
global CLASSIFICATION
CLASSIFICATION = 4

#   学习率
global LEARN_RATE
LEARN_RATE = 0.0001

#   测试集来源
global TEST
TEST = 'train'

#   提前结束的Patience
global PATIENCE
PATIENCE = 0.2*EPOCH

#   使用同一数据集进行训练和测试时
global TEST_SIZE
TEST_SIZE = 0.2
"""
完成训练数据的读取和分词
"""




def load_data_from_one_set(file_word, file_label):
    global SAMPLES
    word_lines = open(file_word, 'r', encoding='utf-8')
    label_lines = open(file_label, 'r', encoding='utf-8')

    sen_lst = []
    sc = 0
    c = SAMPLES
    for line in word_lines:
        words = line.split()
        if 'NULL' in words:
            words.remove('NULL')
        sen_lst.append(words)
        #print(words)
        sc = sc + 1
        if sc >= c:
            break
    SAMPLES = sc
    label_lst = []
    lc = 0
    for line in label_lines:
        lb = int(line)
        label_lst.append(lb)
        #print(lb)
        lc = lc + 1
        if lc >= sc:
            break
    label_lst = to_categorical(label_lst)
    #label_lst = np.asarray(label_lst)
    print(label_lst)
    print(label_lst.shape)
    return sen_lst, label_lst

def load_data_from_different_sets(file_word_train, file_label_train, file_word_test, file_label_test):
    train_words, train_labels = load_data_from_one_set(file_word_train, file_label_train)
    test_words, test_labels = load_data_from_one_set(file_word_test, file_label_test)

    return  train_words, train_labels, test_words, test_labels


def get_embedding_weights(index_dict,word_vectors):
    global WORD_VEC_DIM
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    print("n_symble:" + str(n_symbols))
    embedding_weights = np.zeros((n_symbols, WORD_VEC_DIM))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]

    return n_symbols, embedding_weights


def get_data_from_one_set(file_word, file_label):
    global TEST_SIZE
    #print(x_train.shape[0],y_train.shape[0])
    combined, y = load_data_from_one_set(file_word, file_label)
    model_path = 'my_w2v/s=' + str(WORD_VEC_DIM) + '_mc=' + str(MIN_COUNT) + '_w=' + str(WINDOW) + '_w2v_model.pkl'
    print("word2vec model:" + model_path)
    index_dict, word_vectors, combined_list = word2vec_load(combined, model_path)
    combined = combined_list[0]
    print("！！！！！！！！！！！！！！！！！get_data_from_one_set:")
    print(combined.shape)
    n_symbols, embedding_weights = get_embedding_weights(index_dict, word_vectors)
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=TEST_SIZE)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


def get_data_from_diferrent_set(file_word_train, file_label_train, file_word_test, file_label_test):
    global TEST_SIZE
    #print(x_train.shape[0],y_train.shape[0])
    train_words, train_labels, test_words, test_labels = load_data_from_different_sets(file_word_train,
                                                                                       file_label_train,
                                                                                       file_word_test,
                                                                                       file_label_test)
    model_path = 'my_w2v/s=' + str(WORD_VEC_DIM) + '_mc=' + str(MIN_COUNT) + '_w=' + str(WINDOW) + '_w2v_model.pkl'
    print("word2vec model:" + model_path)
    index_dict, word_vectors, combined_list = word2vec_load(sentences=train_words, model_path=model_path,
                                                            sentences2=test_words)
    n_symbols, embedding_weights = get_embedding_weights(index_dict, word_vectors)
    x_train = combined_list[0]
    x_test = combined_list[1]
    y_train = train_labels
    y_test = test_labels
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


#   ######定义基础LSTM
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    global CLASSIFICATION, BATCH_SIZE, EPOCH, SAMPLES, TEST, WORD_VEC_DIM, MAX_LEN, PATIENCE
    print('Defining a Simple Keras Model...')

    es = EarlyStopping(monitor='val_acc', patience=PATIENCE)
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=WORD_VEC_DIM,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=MAX_LEN))  # Adding Input Length
    model.add(Dropout(0.5))
    model.add(LSTM(output_dim=50))
    model.add(Dropout(0.3))
    model.add(Dense(CLASSIFICATION))
    model.add(Activation('sigmoid'))
    print(model.summary())
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    checkpoint_filepath = 'models-LSTM/model-epoch/model-ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}' \
                          '-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    metrics = Metrics()
    print("LSTM Train...")
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH,
              verbose=1, callbacks=[metrics, checkpoint, es], validation_data=(x_test, y_test))
    fig = 'models-LSTM/results/'+TEST+'_b='+str(BATCH_SIZE)+'_e='+str(EPOCH)+\
          '_w='+str(WORD_VEC_DIM)+'_s='+str(SAMPLES)+'.png'
    xls = 'models-LSTM/results/' + TEST + '_b=' + str(BATCH_SIZE) + \
          '_e=' + str(EPOCH) + '_w=' + str(WORD_VEC_DIM) + '_s=' + str(SAMPLES) + '.xls'
    show_result(history, metrics, fig, xls)
    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

    yaml_string = model.to_yaml()
    with open('models-LSTM/model-yaml/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('models-LSTM/model-yaml/lstm.h5')
    print('Test score:', score)


#   #####定义Bi-LSTM
def train_bilstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    global WORD_VEC_DIM, MAX_LEN, CLASSIFICATION, BATCH_SIZE, EPOCH, TEST, SAMPLES, PATIENCE
    es = EarlyStopping(monitor='val_acc', patience=PATIENCE)
    inputs = Input(shape=(MAX_LEN,), name='my_input_layer')
    embedding = Embedding(output_dim=WORD_VEC_DIM, input_dim=n_symbols,
                          mask_zero=False, weights=[embedding_weights],
                          input_length=MAX_LEN, name="my_embedding_layer"
                          )(inputs)
    drop1 = Dropout(0.3)(embedding)
    lstm_out = Bidirectional(LSTM(WORD_VEC_DIM, return_sequences=True), name='bilstm')(drop1)
    flatten = Flatten()(lstm_out)
    drop2 = Dropout(0.3)(flatten)
    output = Dense(CLASSIFICATION, activation='sigmoid')(drop2)
    model = Model(inputs=inputs, outputs=output)
    print(model.summary())
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    checkpoint_filepath = 'models-BiLSTM/model-epoch/model-ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}' \
                          '-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    metrics = Metrics()
    print("BiLSTM Train...")
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH,
                        verbose=1, callbacks=[metrics, checkpoint, es], validation_data=(x_test, y_test))
    fig = 'models-BiLSTM/results/'+TEST+'_b=' + str(BATCH_SIZE) + '_e=' + str(EPOCH) + \
          '_w=' + str(WORD_VEC_DIM) + '_s=' + str(SAMPLES) + '.png'
    xls = 'models-BiLSTM/results/' + TEST + '_b=' + str(BATCH_SIZE) + \
          '_e=' + str(EPOCH) + '_w=' + str(WORD_VEC_DIM) + '_s=' + str(SAMPLES) + '.xls'
    show_result(history, metrics, fig, xls)
    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

    yaml_string = model.to_yaml()
    with open('models-BiLSTM/model-yaml/bilstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('models-BiLSTM/model-yaml/bilstm.h5')
    print('Test score:', score)


#   ####attention机制
def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)

    a = Dense(MAX_LEN, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


#   ####定义Attentioned-BiLSTM

def train_attentioned_bilstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    global WORD_VEC_DIM, MAX_LEN, CLASSIFICATION, BATCH_SIZE, EPOCH, TEST, SAMPLES, PATIENCE
    es = EarlyStopping(monitor='val_acc', patience=PATIENCE)
    inputs = Input(shape=(MAX_LEN,), name='my_input_layer')
    embedding = Embedding(output_dim=WORD_VEC_DIM, input_dim=n_symbols,
                          mask_zero=False, weights=[embedding_weights],
                          input_length=MAX_LEN, name="my_embedding_layer"
                          )(inputs)
    drop1 = Dropout(0.3)(embedding)

    lstm_out = Bidirectional(LSTM(WORD_VEC_DIM, return_sequences=True), name='bilstm')(drop1)
    attention_mul = attention_3d_block(lstm_out)
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(0.3)(attention_flatten)
    output = Dense(CLASSIFICATION, activation='sigmoid')(drop2)
    model = Model(inputs=inputs, outputs=output)
    print(model.summary())
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    checkpoint_filepath = 'models-Attentioned-BiLSTM/model-epoch/model-ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}' \
                          '-val_loss{val_loss:.3f}-val_acc{val_acc:.3f}.h5'
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    metrics = Metrics()
    print("Attentioned-BiLSTM Train...")
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCH,
                        verbose=1, callbacks=[metrics, checkpoint, es], validation_data=(x_test, y_test))
    fig = 'models-Attentioned-BiLSTM/results/'+TEST+'_b=' + str(BATCH_SIZE) + \
          '_e=' + str(EPOCH) + '_w=' + str(WORD_VEC_DIM) + '_s=' + str(SAMPLES) + '.png'
    xls = 'models-Attentioned-BiLSTM/results/'+TEST+'_b=' + str(BATCH_SIZE) + \
          '_e=' + str(EPOCH) + '_w=' + str(WORD_VEC_DIM) + '_s=' + str(SAMPLES) + '.xls'
    show_result(history, metrics, fig, xls)
    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

    yaml_string = model.to_yaml()
    with open('models-Attentioned-BiLSTM/model-yaml/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('models-Attentioned-BiLSTM/model-yaml/lstm.h5')
    print('Test score:', score)


#   训练模型，并保存


def train():
    print('Loading Data...')

    #print(len(combined),len(y))
    print('Tokenising...')
    #combined = tokenizer(combined)
    print('Training a Word2vec model...')

    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data_from_one_set('train_words_media.txt',
                                                                                           'train_sentiment_media.txt')
    #n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data_from_diferrent_set('train_words_media.txt',
     #                                                                                            'train_sentiment_media.txt',
      #                                                                                           'train_words_media.txt',
       #                                                                                          'train_sentiment_media.txt',
        #                                                                                         )
    #print(x_train.shape,y_train.shape)
    #train_attentioned_bilstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)
    train_attentioned_bilstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)
    #train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    train()
