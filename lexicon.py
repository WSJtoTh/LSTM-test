#!/usr/bin/env/ python
#_*_ coding:utf-8_*_
from myclass import SentiWordNet, SentiLexicon, SentiInRam
import nltk
import gensim.models
global My_SentiLexicon
My_SentiLexicon = SentiLexicon()

global My_Senti_Lex
My_Senti_Lex = SentiInRam('CleanSWN.txt')

#global My_Senti_Lex
#My_Senti_Lex = SentiInRam('CleanSWN.txt')

def build_emoji_lexicon():
    emoji_file = open("D:\pyCharm\LSTM\emoji/emoji_joined.txt", 'r', encoding="utf-8")
    lines = emoji_file.readlines()
    #e2v = gensim.models.KeyedVectors.load_word2vec_format('D:\pyCharm\LSTM\emoji2vec\pre-trained/emoji2vec.bin', binary=True)
    #print("emoji")


def deal_sentiwordnet():
    sentilexicon = SentiLexicon()
    swn_file = open('SentiWordNet.txt', 'r', encoding="utf-8")
    lines = swn_file.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        else:
            info = line.split("\t")
            s = SentiWordNet(info[0], info[1], info[2], info[3], info[4])
            sentilexicon.senti_lexicon.append(s)

    print("finish")
    swn_file.close()
    clean_swn_filw = open("CleanSWN.txt", 'a')
    sentilexicon.write_into_file(clean_swn_filw)


def read_in_clean_lexicon():
    global My_Senti_Lex

    exist, pos_score, neg_score = My_Senti_Lex.get_score(pos='a', word='bad')
    if exist:
        score = pos_score - neg_score
        print(str(score))
    else:
        print("the word is not in dict")


def merge_into_phrase(words):
    phrase_len = 9
    while len(words) >= 2 and phrase_len >= 2:
        # print(words)
        if phrase_len <= len(words):
            ngram = list(nltk.ngrams(words, phrase_len))
            # print(ngram)
            if len(ngram) == 0:
                print("0000000000000000000000000000000000000000000000")
            index = 0
            skip = 0
            for gram in ngram:
                if skip > 0:
                    skip = skip - 1
                    continue
                else:
                    ph = ''
                    for g in gram:
                        if ph == '':
                            ph = g
                        else:
                            ph = ph + '_' + g
                    flag = My_Senti_Lex.phrase_determine(phrase=ph, length=phrase_len)
                    if flag:
                        words[index] = ph
                        i = 1
                        #print(ph)
                        #   print(words)
                        while i < phrase_len:
                            #print('i='+str(i))
                            #print(str(index))
                            #print(len(words))
                            words.remove(words[index + 1])
                            i = i + 1
                        skip = phrase_len - 1
                    else:
                        skip = 0
                    index = index + 1
        phrase_len = phrase_len - 1

    return words


def find_phrase(filename, filename1):
    global  My_Senti_Lex
    file = open(filename, 'r', encoding='utf-8')
    new_file = open(filename1, 'w', encoding='utf-8')
    lines = file.readlines()
    phrase_len = 2
    for line in lines:
        line = line.split('\n')
        sen = line[0].split('\t')
        print(sen)
        print(len(sen))
        if '' in sen:
            sen.remove('')
        print(sen)
        print(len(sen))
        sen_count = 0
        for s in sen:
            sen_count = sen_count + 1
            words = s.split(' ')
            words = merge_into_phrase(words)
            """
            if len(words) < phrase_len:
                print(words)
                print(len(words))
                for word in words:
                    if sen_count == 3:
                        new_file.write(word)
                    else:
                        new_file.write(word+'\t')
                continue
            else:
                #print(words)
                ngram = list(nltk.ngrams(words, phrase_len))
                #print(ngram)
                if len(ngram) == 0:
                    print("00000")
                    continue
                index = 0
                skip = 0
                for gram in ngram:
                    if skip > 0:
                        skip = skip - 1
                        continue
                    else:
                        ph = ''
                        for g in gram:
                            if ph == '':
                                ph = g
                            else:
                                ph = ph + '_' + g
                        #print(ph)
                        flag = My_Senti_Lex.phrase_determine(phrase=ph, length=phrase_len)
                        if flag:
                            #print("cunzai")
                            #print(ph)
                            words[index] = ph
                            i = 1
                            while i < phrase_len:
                                words.remove(words[index+i])
                                i = i + 1
                            #print(words)
                            skip = phrase_len - 1
                            #index = index + 1
                        else:
                            skip = 0
                        index = index + 1
            """
            length_words = len(words)
            j = 0
            # 处理完的单词写入文件中
            for word in words:
                j = j + 1
                if j < length_words:
                    new_file.write(word + " ")
                else:
                    new_file.write(word + "\t")
        new_file.write('\n')





if __name__ == '__main__':
    #deal_sentiwordnet()
   # read_in_clean_lexicon()
    find_phrase('train_words.txt', 'train_words_phrase.txt')
