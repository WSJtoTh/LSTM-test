#!/usr/bin/env/ python
#_*_ coding:utf-8_*_


class Sentiment(object):
    #DEFAULT = 0     #未曾进行分类
    HAPPY = 0
    ANGRY = 1
    SAD = 2
    OTHERS = 3


class Sentences(object):

    def __init__(self):
        #print("开始初始化sentence")
        self.sentence_words = [[], [], []]
        self.sentence_faces = [[], [], []]
        self.sentence_punctuations = [[], [], []]
        self.sentiment = Sentiment.OTHERS

    def modify_words(self, words, i):
        self.sentence_words[i] = words

    def modify_faces(self, faces, i):
        self.sentence_faces[i] = faces

    def modify_puncs(self, puncs, i):
        self.sentence_punctuations[i] = puncs

    def modify_sentiment(self, sentiment):
        self.sentiment = sentiment

    def get_sentiment(self):
        return self.sentiment

    def get_words_onceatime(self, i):
        return self.sentence_words[i]

    def get_faces_onceatime(self, i):
        return self.sentence_faces[i]

    def get_puncs_onceatime(self, i):
        return self.sentence_punctuations[i]

    def get_words(self):
        return self.sentence_words

    def get_faces(self ):
        return self.sentence_faces

    def get_puncs(self):
        return self.sentence_punctuations


class SentenceList(object):
    sentences = []
    sentences_amount = 0

    def __init__(self, num):
        i = 0
        #print("i="+str(i))
        while i < num:
            new_sentence = Sentences()
            self.sentences.append(new_sentence)
            i = i + 1

    def modify_onceatime(self, words, faces, puncs, j, n):
        self.sentences[n].modify_words(words, j)
        self.sentences[n].modify_faces(faces, j)
        self.sentences[n].modify_puncs(puncs, j)

    def modify_sentiment(self, n, sentiment):
        self.sentences[n].modify_sentiment(sentiment)

    def write_into_file(self, word_file, face_file, punctuation_file, sentiment_file, num):       #filename:写入文件的文件句柄，num写入的数据条数
        # 将处理好的单词写入文件中
        n = 0
        for sentence in self.sentences:
            #print("di"+str(n)+"tiaoxinxi")
            #print(sentence)
            #情感极性写入文件中
            sentiment_file.write(str(sentence.get_sentiment())+"\n")
            for i in [0, 1, 2]:
                words = sentence.get_words_onceatime(i)
                length_words = len(words)
                faces = sentence.get_faces_onceatime(i)
                length_face = len(faces)
                puncs = sentence.get_puncs_onceatime(i)
                length_punc = len(puncs)
                j = 0
                #处理完的单词写入文件中
                if length_words > 0:
                    for word in words:
                        j = j + 1
                        if j < length_words:
                            word_file.write(word+" ")
                        else:
                            word_file.write(word+"\t")
                else:
                    word_file.write("NULL\t")
                j = 0
                #处理完的表情写入文件中
                if length_face > 0:
                    for face in faces:
                        j = j + 1
                        if j < length_words:
                            face_file.write(face + " ")
                        else:
                            face_file.write(face + "\t")
                else:
                    face_file.write("NULL\t")
                j = 0
                #处理完的标点符号写入文件中
                if length_punc > 0:
                    for punctuation in puncs:
                        j = j + 1
                        if j < length_punc:
                            punctuation_file.write(punctuation + " ")
                        else:
                            punctuation_file.write(punctuation + "\t")
                else:
                    punctuation_file.write("NULL\t")

                #完成文件写入后，将内存中的暂存变量清空
                sentence.modify_words([], i)
                sentence.modify_faces([], i)
                sentence.modify_puncs([], i)
            word_file.write("\n")
            face_file.write("\n")
            punctuation_file.write("\n")
            n = n + 1
            #这个判断句主要是最后一部分的处理
            if n == num:
                break
        print("完成单词写入文件和内存清空")

    def print_data(self):
        k = 0
        for sentence in self.sentences:
            k = k + 1
            print(sentence)
            for i in [0, 1, 2]:
                words = sentence.get_words_oncestime()
                length_words = len(words)
                faces = sentence.get_faces_onceatime()
                length_face = len(faces)
                puncs = sentence.get_puncs_onceatime()
                length_punc = len(puncs)
                if length_words > 0:
                    for word in words:
                        print(word)
                else:
                    print("NULL")
                if length_face > 0:
                    for face in faces:
                        print(face)
                else:
                    print("NULL")
                if length_punc > 0:
                    for punctuation in puncs:
                        print(punctuation)
                else:
                    print("NULL")

    #def read_from_clean_words_file(self, ):


class Face(object):

    def __init__(self, face, sentiment):
        self.face = face
        self.face_sentiment = {"HAPPY": 0, "SAD": 0, "ANGRY": 0, "OTHERS": 0}    #表情情感计数
        if sentiment == Sentiment.HAPPY:
            self.face_sentiment.update({"HAPPY": 1})
        elif sentiment == Sentiment.ANGRY:
            self.face_sentiment.update({"SAD": 1})
        elif sentiment == Sentiment.SAD:
            self.face_sentiment.update({"ANGRY": 1})
        else:
            self.face_sentiment.update({"OTHERS": 1})
        self.face_frequence = 1

    def show_sentiment(self):
        print("Happy\tSad\tAngry\tOthers")
        print(str(self.face_sentiment["HAPPY"])+"\t"+str(self.face_sentiment["SAD"])+"\t"
              +str(self.face_sentiment["ANGRY"])+"\t"+str(self.face_sentiment["OTHERS"]))

    def add_frequence(self, sentiment):
        if sentiment == Sentiment.HAPPY:
            polarity = "HAPPY"
        elif sentiment == Sentiment.ANGRY:
            polarity = "ANGRY"
        elif sentiment == Sentiment.SAD:
            polarity = "SAD"
        else:
            polarity = "OTHERS"
        old = self.face_sentiment[polarity]
        self.face_sentiment[polarity] = old+1
        self.face_frequence = self.face_frequence + 1


class FaceList(object):
    face = []       #存储表情的名称
    face_list = []      #存储表情的详细信息
    face_amount = 0


class PunctuationList(object):
    punctuation = []
    punctuation_list = []
    punctuation_amount = 0


class SentiWordNet(object):

    """
    对SynsetTerms做处理
    """
    @staticmethod
    def wash_terms(terms):
        temps = terms.split(" ")
        SentiLexicon.count_words(len(temps))
        words = []
        for t in temps:
            t = t.split("#")
            words.append(t[0])
        return words

    """
    初始化
    """
    def __init__(self, pos, id, pos_score, neg_score, terms):
        self.pos = pos
        self.id = id
        self.pos_score = pos_score
        self.neg_score = neg_score
        self.terms = self.wash_terms(terms)

    def get_pos(self):
        return self.pos

    def get_id(self):
        return self.id

    def get_pos_score(self):
        return self.pos_score

    def get_neg_score(self):
        return self.neg_score

    def get_words(self):
        return self.terms


class SentiLexicon(object):
    senti_lexicon = []
    word_amount = 0

    """
    情感词典数量纪录
    """
    @staticmethod
    def count_words(num):
        SentiLexicon.word_amount = SentiLexicon.word_amount + num

    def write_into_file(self, file_handle):
        file_handle.write("#lexicon_amount:" + str(self.word_amount) + "\n")
        for words in self.senti_lexicon:
            file_handle.write(str(words.get_pos()) + "\t" + str(words.get_id()) + "\t" + str(words.get_pos_score())
                              + "\t" + str(words.get_neg_score()) + "\t")
            for w in words.get_words():
                file_handle.write(w + "%")
            file_handle.write("\n")


"""
放在内存中的情感词典，便于遍历
"""


class SentiInRam(object):
    #形容词
    a_words_list = []
    a_id_list = []
    a_pos_score = []
    a_neg_score = []
    #副词
    r_words_list = []
    r_id_list = []
    r_pos_score = []
    r_neg_score = []
    #名词
    n_words_list = []
    n_id_list = []
    n_pos_score = []
    n_neg_score = []
    #动词
    v_words_list = []
    v_id_list = []
    v_pos_score = []
    v_neg_score = []

    #2词短语
    phrase_2 = {}
    #3词短语
    phrase_3 = {}
    #4词短语
    phrase_4 = {}
    #5词短语
    phrase_5 = {}
    #6词短语
    phrase_6 = {}
    phrases = [{}, {}, {}, {}, {}, {}, {}, {}]


    @staticmethod
    def deal_phrase(words, pos, list_index):
        for w in words:
            if '-' in w:
                index = words.index(w)
                w = w.replace('-', '_')
                words[index] = w
            if '_' in w:
                tmp = w.split('_')
                #build_phrase_list(tmp, len(tmp), 'a', len)
                i = len(tmp) - 2
                #new_phrase = {w: [pos, list_index]}
                #print(type(SentiInRam.phrases[i]))
                SentiInRam.phrases[i][w] = [pos, list_index]

        return words

    def __init__(self, filename):
        lines = open(filename, 'r')
        for line in lines:
            if line.startswith('#'):
                continue
            else:
                line = line.split('\t')
                words = line[4].split('%')
                words.remove('\n')
                #print(words)
                if line[0] == 'a':
                    words = self.deal_phrase(words, 'a', len(self.a_words_list))
                    self.a_words_list.append(words)
                    self.a_id_list.append(line[1])
                    self.a_pos_score.append(line[2])
                    self.a_neg_score.append(line[3])
                elif line[0] == 'n':
                    words = self.deal_phrase(words, 'n', len(self.n_words_list))
                    self.n_words_list.append(words)
                    self.n_id_list.append(line[1])
                    self.n_pos_score.append(line[2])
                    self.n_neg_score.append(line[3])
                elif line[0] == 'v':
                    words = self.deal_phrase(words, 'v', len(self.v_words_list))
                    self.v_words_list.append(words)
                    self.v_id_list.append(line[1])
                    self.v_pos_score.append(line[2])
                    self.v_neg_score.append(line[3])
                elif line[0] == 'r':
                    words = self.deal_phrase(words, 'r', len(self.r_words_list))
                    self.r_words_list.append(words)
                    self.r_id_list.append(line[1])
                    self.r_pos_score.append(line[2])
                    self.r_neg_score.append(line[3])
                else:
                    print("no such kind")
        for ps in self.phrases:
            ##写进一个文件中
            print(ps)

    @staticmethod
    def record_new_word(pos, pos_score, neg_score, word):
        new_word = open('new_word.txt', 'a', encoding='utf-8')
        new_word.write(pos + '\t' + pos_score + '\t' + neg_score + '\t' + word + '\n')
        new_word.close()

    def get_score(self, pos, word):
        exist = False
        pos_score = 0
        neg_score = 0
        if pos == 'a':
            #print('a')
            for wl in self.a_words_list:
                if word in wl:
                    print(wl)
                    index = self.a_words_list.index(wl)
                    #print(str(index))
                    pos_score = float(self.a_pos_score[index])
                    #print(str(pos_score))
                    neg_score = float(self.a_neg_score[index])
                    #print(str(neg_score))
                    exist = True
                    #print(exist)
                    break
        elif pos == 'n':
            for wl in self.n_words_list:
                if word in wl:
                    index = self.n_words_list.index(wl)
                    pos_score = float(self.n_pos_score[index])
                    #print(str(pos_score))
                    neg_score = float(self.n_neg_score[index])
                    exist = True
                    break
        elif pos == 'r':
            for wl in self.r_words_list:
                if word in wl:
                    index = self.r_words_list.index(wl)
                    pos_score = float(self.r_pos_score[index])
                    neg_score = float(self.r_neg_score[index])
                    exist = True
                    break
        elif pos == 'v':
            for wl in self.v_words_list:
                if word in wl:
                    index = self.v_words_list.index(wl)
                    pos_score = float(self.v_pos_score[index])
                    neg_score = float(self.v_neg_score[index])
                    exist = True
                    break
        else:
            print("没有这种词性")
        #print(exist)
        if exist == False:
            print("开始调用记录新词汇的函数")
            self.record_new_word(pos, pos_score, neg_score, word)
        #print(type(pos_score))
        return exist, pos_score, neg_score

    def phrase_determine(self, phrase, length):
        i = length - 2
        pl = self.phrases[i]
        flag = False
        #print(phrase)
        if phrase in pl.keys():
            #print("!!!11!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            flag = True

        return flag


class ReplaceLexicon(object):
    raw_list = []
    replace_list = []
    replace_amount = 0

    def __init__(self, file):
        lines = file.readlines()
        for line in lines:
            temp = line.split('\t')
            print(temp)
            self.raw_list.append(temp[0])
            temp[1] = temp[1].split('\n')[0]
            self.replace_list.append(temp[1])
            print(temp)
            self.replace_amount = self.replace_amount + 1

    def replace_word(self, words):
        for word in words:
            if word in self.raw_list:
                i = self.raw_list.index(word)
                j = words.index(word)
                words[j] = self.replace_list[i]

        return words





