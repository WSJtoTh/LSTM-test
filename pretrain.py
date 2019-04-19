#!/usr/bin/env/ python
#_*_ coding:utf-8_*_
import chardet
import re
from nltk.tokenize import WordPunctTokenizer
from myclass import Sentences, Sentiment,SentiInRam
import nltk
#nltk.download('stopwords')
from myclass import Face, FaceList, PunctuationList, SentenceList, ReplaceLexicon
from lexicon import merge_into_phrase
global My_FaceList
My_FaceList = FaceList()
global My_PunctuationList
My_PunctuationList = PunctuationList()
global MEMORY_NUM
MEMORY_NUM = 100
print("???")
global My_Sentences
My_Sentences = SentenceList(MEMORY_NUM)
global StopWordSet
StopWordSet = nltk.corpus.stopwords.words('english')
replace_file = open('abbreviation.txt', 'r', encoding='utf-8')
global My_ReplaceList
My_ReplaceList = ReplaceLexicon(replace_file)
#print(StopWordSet)
global My_Senti_Lex
My_Senti_Lex = SentiInRam('CleanSWN.txt')

def restore_words(words):
    lem = nltk.WordNetLemmatizer()
    for word in words:
        print("word:"+word)
        temp = lem.lemmatize(word)
        print(temp)
"""
加入表情、符号
"""


def add_face(newFace, sentiment, kind):

    if kind == 1:
        for face in newFace:
            global My_FaceList
            if face in My_FaceList.face:
                loc = My_FaceList.face.index(face)
                My_FaceList.face_list[loc].add_frequence(sentiment)
            else:
                new_face = Face(face=face, sentiment=sentiment)
                My_FaceList.face.append(face)
                My_FaceList.face_list.append(new_face)
                My_FaceList.face_amount = My_FaceList.face_amount + 1
    else:
        global My_PunctuationList
        if newFace in My_PunctuationList.punctuation:
            idx = My_PunctuationList.punctuation.index(newFace)
            My_PunctuationList.punctuation_list[idx].add_frequence(sentiment)
        else:
            new_face = Face(face=newFace, sentiment=sentiment)
            My_PunctuationList.punctuation.append(newFace)
            My_PunctuationList.punctuation_list.append(new_face)
            My_PunctuationList.punctuation_amount = My_PunctuationList.punctuation_amount + 1


"""
将表情、符号、单词进行分离
"""


def split_punctuation_face(word_list, sentiment):
    pattern_abbreviation = re.compile(r'(\w+\'\w+)|([!@#$%^&*()_+\-=`~{}\[\]:;\'\"<>,\.|\\]+)')      #匹配缩写（don't或warm_heart或word-word）
    word_seperate = []
    for word in word_list:      #英文缩写不进行切分
        result = pattern_abbreviation.match(word)
        if result:
            word_seperate.append(word)
        else:
            word_split = WordPunctTokenizer().tokenize(word)
            for w in word_split:
                word_seperate.append(w)
    face_list = []
    punctuation_list = []
    words = []
    pattern = re.compile(r'[^A-Za-z0-9]+')
    pattern_punctuation = re.compile(r'[!@#$%^&*()_+\-=`~{}\[\]:;\'\"<>,\.|\\?/]+')
    for word in word_seperate:      #完成表情和符号的切分
        result1 = re.match(pattern, word)
        if result1:     #判断是否为非字母
            result2 = re.match(pattern_punctuation, word)
            if result2:
                #print("得到一个符号表情" + word)
                add_face(word, sentiment, 2)
                punctuation_list.append(word)
            else:
                ##print("表情" + word)
                add_face(word, sentiment, 1)
                face_list.append(word)
        else:
            words.append(word)
            #着两行完成了去停用词，但可能效果不是很好
            #global StopWordSet
            #if word not in StopWordSet:

    return words, punctuation_list, face_list


"""
判断情感极性
"""


def sentiment_polarity(sentiment):
    #print(sentiment)
    #print(type(sentiment))
    if sentiment == "happy\n":
        result = Sentiment.HAPPY
    elif sentiment == "sad\n":
        result = Sentiment.SAD
    elif sentiment == "angry\n":
        result = Sentiment.ANGRY
    else:
        result = Sentiment.OTHERS

    return result


"""
完成词性分类
"""


def sort_pos(tag):
    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
        pos = 'n'
    elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
        pos = 'v'
    elif tag in ['RB', 'RBR', 'RBS']:
        pos = 'r'
    elif tag in ['JJ', 'JJR', 'JJS']:
        pos = 'a'
    else:
        pos = 'o'

    return pos


"""
对单词进行归一化处理,词形转换
"""


def my_lemmatizer(words_temp):
    word_pos = nltk.pos_tag(words_temp)
    word_list = []
    #print(word_pos)
    wordnet_lemmatizer = nltk.WordNetLemmatizer()
    for word in word_pos:
        #print(word)
        pos = sort_pos(word[1])
        if pos == 'o':
            word_list.append(word[0])
        else:
            word_list.append(wordnet_lemmatizer.lemmatize(word[0], pos=pos))
    return word_list


"""
去停用词
"""


def delete_stopwords(words):
    global StopWordSet
    for word in words:
        if word in StopWordSet:
            words.remove(word)

    return words


"""
数据清洗
"""


def data_wash(text, n):
    global My_Sentences, My_ReplaceList
    s = Sentences()
    sentences = text.split('\t')
    del sentences[0]
    if len(sentences) >= 4:
        #print("带标签数据")
        s.sentiment = sentiment_polarity(sentences[3])
        #print(str(s.sentiment))
        #print(sentences[3])
        del sentences[3]

    #print("在datawash中+\n进行数据处理前n="+str(n))
    j = 0
    for sentence in sentences:
        sentence = sentence.lower()
        #初步将句子切分成单词
        words_temp = sentence.split(' ')
        #   除去列表中的空字符串，否则会报错
        while '' in words_temp:
            words_temp.remove('')
        #print(words_temp)
        #缩略语、俚语替换
        words_temp = My_ReplaceList.replace_word(words_temp)
        #符号和表情切分
        words, punctuations, faces = split_punctuation_face(words_temp, s.sentiment)
        #词形变换
        words = my_lemmatizer(words)
        #合成短语
        words = merge_into_phrase(words)
        #去停用词
        words = delete_stopwords(words)
        My_Sentences.modify_onceatime(words, faces, punctuations, j, n)
        j = j + 1
    My_Sentences.modify_sentiment(n, s.sentiment)
    My_Sentences.sentences_amount = My_Sentences.sentences_amount + 1


"""
打印表情
"""


def print_facelist():
    global My_FaceList
    print("当前表情列表中有"+str(My_FaceList.face_amount)+"个表情")
    print("表情\t出现总数")
    for face in My_FaceList.face_list:
        print(face.face+"\t"+ str(face.face_frequence))
        face.show_sentiment()


"""
打印符号
"""


def print_punctuationlist():
    global My_PunctuationList
    print("当前符号列表中有"+str(My_PunctuationList.punctuation_amount)+"个符号")
    print("表情\t出现总数")
    for punc in My_PunctuationList.punctuation_list:
        print(punc.face + "\t" + str(punc.face_frequence))
        punc.show_sentiment()


"""
d打印句子
"""


def print_sentences():
    global My_Sentences
    i = 0
    print("开始打印列表")
    for sentence in My_Sentences.sentences:
        print(sentence)
        i = i + 1
        print("第" + str(i) + "条信息")
        for j in [0, 1, 2]:
            print("第"+str(j+1)+"句话\nwords:")
            for word in sentence.sentence_words[j]:
                print(word)
            print("face:")
            for face in sentence.sentence_faces[j]:
                print(face)
            print("punctuation:")
            for punc in sentence.sentence_punctuations[j]:
                print(punc)



if __name__ == '__main__':
    trainFile = open("train.txt", encoding='utf-8')
    wordFile = open("train_words_media.txt", 'w', encoding="utf-8")
    faceFile = open("train_faces_media.txt", 'w', encoding="utf-8")
    puncFile = open("train_puncs_media.txt", 'w', encoding="utf-8")
    sentFile = open("train_sentiment_media.txt", 'w', encoding="utf-8")
    lines = trainFile.readlines()
    length = len(lines)
    amount = 0
    print("已经完成文件打开")
    count = 0
    try:
        for line in lines:
            data_wash(line, count)
            count = count + 1
            amount = amount + 1
            if amount == length:
                #print("amount="+str(amount))
                My_Sentences.write_into_file(wordFile, faceFile, puncFile, sentFile, count)
                break
            if count >= MEMORY_NUM:
                #print("count=" + str(count))
                count = count % MEMORY_NUM
                My_Sentences.write_into_file(wordFile, faceFile, puncFile, sentFile, MEMORY_NUM)

    finally:
        print(My_Sentences)
        print_facelist()
        print_punctuationlist()
        trainFile.close()
        wordFile.close()
        faceFile.close()
        puncFile.close()




