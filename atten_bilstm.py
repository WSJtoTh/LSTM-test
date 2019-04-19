#!/usr/bin/env/ python
#_*_ coding:utf-8_*_#!/usr/bin/env/ python
#_*_ coding:utf-8_*_

from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Permute, multiply, Input
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

def encode_label_2D(file_label):
    file = open(file_label, 'r', encoding='utf-8')
    labels = file.readlines()
    #label_lst = []
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

    #for e in result:
      #  print(e)
    print(result[0].shape)
    return result


def load_data_3D(file_word, file_text, file_label):
    data_dict = {}
    lable_lst = encode_label_2D(file_label)
    #lable_lst = label_int(file_label)
    #获取词向量
    word_vec = open(file_word, 'r', encoding='utf-8')
    #label = open(file_label, 'r', encoding='utf-8')
    for line in word_vec:
        values = line.split()
        word = values[0]
        #vec = np.asarray(values[1:], dtype='float32')
        vec = []
        for v in values[1:]:
            vec.append(v)
        #print(vec.shape)
        #print(vec)
        data_dict[word] = vec
    word_vec.close()
    print('load %s word vectors' % len(data_dict))

    #得到句子
    global MAX_LEN
    text = open(file_text, 'r', encoding='utf-8')
    sen_lst = []
    max_len = 0
    ze = ['0']*100
    tmp = 0
    for line in text:
        words = line.split()
        c = 0
        #sen = []
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
            #sen = sen + ze

        #print(str(len(sen)))
        #print(type(sen))

        #sen = sen.reshape(((tmp-)*))
        #sen_lst.append(sen)
        tmp = tmp + 1
        if tmp > 25:
            break
    #返回句子列表和标签列表
    #sen_lst = np.reshape(sen_lst, (len(sen_lst), 1, 1))
    #sen_lst = np.reshape(sen_lst, (len(sen_lst), 100, 1))
    #for i in sen_lst:
     #   print(i)
    #sen_lst = np.mat(sen_lst)
    #print(sen_lst[0])
    #print(sen_lst.shape)
    #sen_lst = sen_lst.reshape((tmp-1, MAX_LEN, WORD_VEC_DIM))
    #print(sen_lst.shape)
    sen_lst = np.asarray(sen_lst)
    sen_lst = sen_lst.reshape((tmp, MAX_LEN, WORD_VEC_DIM))
    print(sen_lst.shape)
    print(sen_lst[0].shape)
    return sen_lst[2000:10000], lable_lst[2000:10000], sen_lst[:2000], lable_lst[:2000]


"""
完成训练数据的读取和分词
"""


def load_data(file_word, file_label):
    word_lines = open(file_word, 'r', encoding='utf-8')
    #label_lines = open(file_label, 'r', encoding='utf-8')
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
    return sen_lst, label_lst, c

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


def attention_3d_block(inputs):
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)

    a = Dense(MAX_LEN, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul
"""
def train_attention_lstm(n_samples, n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    global WORD_VEC_DIM, MAX_LEN
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=WORD_VEC_DIM,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=MAX_LEN))  # Adding Input Length
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(output_dim=50, return_sequences=True), name='bilstm'))
    model.add(Permute((2, 1)))
    model.add(Dense(MAX_LEN, activation='softmax'))
    model.add(Permute((2, 1), name='attention_vec'))
    
    model.add(Dropout(0.3))
    model.add(Dense(5))
    model.add(Activation('sigmoid'))
    print(model.summary())
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    batch_size = 1
    n_epoch = 20
    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('lstm.h5')
    print('Test score:', score)


"""


##定义网络结构
def train_attentioned_lstm(n_samples, n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    global WORD_VEC_DIM, MAX_LEN
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
    output = Dense(5, activation='sigmoid')(drop2)
    model = Model(inputs=inputs, outputs=output)
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
    combined, y, n_samples = load_data('train_words_media.txt', 'train_sentiment_media.txt')
    #print(len(combined),len(y))
    print('Tokenising...')
    #combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors,combined = word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    #print(x_train.shape,y_train.shape)
    #x_train, y_train, x_test, y_test = load_data_3D('word_vec.txt', 'train_words_media.txt', 'train_sentiment_media.txt')
    train_lstm(n_samples, n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    train()