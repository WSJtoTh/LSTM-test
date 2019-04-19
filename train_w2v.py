#!/usr/bin/env/ python
#_*_ coding:utf-8_*_
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
global MAX_LEN
MAX_LEN = 158
#   词向量维度
global WORD_VEC_DIM
WORD_VEC_DIM = 250

#   最小词频
global MIN_COUNT
MIN_COUNT = 1

#   训练数据的数据量
global SAMPLES
SAMPLES = 1000000

#   滑动窗口
global WINDOW
WINDOW = 5


from keras.preprocessing import sequence


def load_data(file_word):
    word_lines = open(file_word, 'r', encoding='utf-8')

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

    return sen_lst


def create_dictionaries(model=None, sen_lst=None, sen_lst_2=None):
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

        combined = []
        sen_lst = parse_dataset(sen_lst)
        global MAX_LEN
        print(len(sen_lst))
        sen_lst = sequence.pad_sequences(sen_lst, maxlen=MAX_LEN)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        print(sen_lst.shape)
        combined.append(sen_lst)
        if sen_lst_2 is not None:
            sen_lst_2 = parse_dataset(sen_lst_2)
            sen_lst_2 = sequence.pad_sequences(sen_lst_2, maxlen=MAX_LEN)
            print(sen_lst_2.shape)
            combined.append(sen_lst_2)
        print("###############################create_dictionary:")
        #print(combined.shape)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined, model_path):
    #model = Word2Vec.load('D:\pyCharm\LSTM\word2vec.model')
    global MIN_COUNT, WINDOW, WORD_VEC_DIM
    model = Word2Vec(size=WORD_VEC_DIM,
                     min_count=MIN_COUNT,
                     window=WINDOW,
                     workers=1)
    model.build_vocab(combined)
    model.train(combined, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_path)
    #index_dict, word_vectors, combined = create_dictionaries(model=model, sen_lst=combined)
    #return index_dict, word_vectors, combined


def word2vec_load(sentences, model_path, sentences2=None):
    model = Word2Vec.load(model_path)
    if sentences2 is None:
        index_dict, word_vectors, sen_lst = create_dictionaries(model=model, sen_lst=sentences)
    else:
        index_dict, word_vectors, sen_lst = create_dictionaries(model=model, sen_lst=sentences, sen_lst_2=sentences2)
    return index_dict, word_vectors, sen_lst


if __name__ == '__main__':
    model_path = 'my_w2v/s='+str(WORD_VEC_DIM)+'_mc='+str(MIN_COUNT)+'_w='+str(WINDOW)+'_w2v_model.pkl'
    sentences = load_data('train_words_media.txt')
    word2vec_train(sentences, model_path)
