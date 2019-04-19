#!/usr/bin/env/ python
#_*_ coding:utf-8_*_

from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Permute
#from keras import wrappers
#from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Bidirectional, Dropout, Flatten, Embedding
#from keras.layers import Masking
from keras.utils import to_categorical
import yaml
#句子的最大长度
global MAX_LEN
MAX_LEN = 158

#词向量的维数
global WORD_VEC_DIM
WORD_VEC_DIM = 100

#训练数据的数据量
global SAMPLES
SAMPLES = 15



#训练数据的数据量
global SAMPLES
SAMPLES = 100000

#训练Batch
global BATCH_SIZE
BATCH_SIZE = 50

#epoch
global EPOCH
EPOCH = 100

#分类器
global CLASSIFICATION
CLASSIFICATION = 4

#学习率
global LEARN_RATE
LEARN_RATE = 0.0001


"""
完成训练数据的读取和分词
"""


def load_data(file_word, file_label):
    word_lines = open(file_word, 'r', encoding='utf-8')
    label_lines = open(file_label, 'r', encoding='utf-8')

    sen_lst = []
    sc = 0
    c = 10000
    for line in word_lines:
        words = line.split()
        if 'NULL' in words:
            words.remove('NULL')
        sen_lst.append(words)
        #print(words)
        sc = sc + 1
        if sc >= c:
            break

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

def create_dictionaries(model=None,sen_lst=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (sen_lst is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(sen_lst):
            ''' Words become integers
            '''
            data=[]
            for sentence in sen_lst:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined = parse_dataset(sen_lst)
        global MAX_LEN
        combined = sequence.pad_sequences(combined, maxlen=MAX_LEN)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    #model = Word2Vec.load('D:\pyCharm\LSTM\word2vec.model')
    model = Word2Vec(size=100,
                     min_count=1,
                     window=5,
                     workers=1)
    model.build_vocab(combined)
    model.train(combined, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, sen_lst=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict,word_vectors,combined,y):
    global WORD_VEC_DIM
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    print("n_symble:"+str(n_symbols))
    embedding_weights = np.zeros((n_symbols, WORD_VEC_DIM))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    #print(x_train.shape[0],y_train.shape[0])
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    global WORD_VEC_DIM, MAX_LEN
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=WORD_VEC_DIM,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=MAX_LEN))  # Adding Input Length
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(output_dim=50, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Dense(4))
    model.add(Activation('sigmoid'))
    print(model.summary())
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    batch_size = 1
    n_epoch = 20
    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('lstm.h5')
    print('Test score:', score)


#   训练模型，并保存


def train():
    print('Loading Data...')
    combined, y =load_data('train_words_media.txt', 'train_sentiment_media.txt')
    print(len(combined),len(y))
    print('Tokenising...')
    #combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    #print(x_train.shape,y_train.shape)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)


if __name__ == '__main__':
    train()