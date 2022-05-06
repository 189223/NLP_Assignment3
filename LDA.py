#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Xu
# @date 2022/5/5
# @file LDA.py
from itertools import tee

import gensim.models
import jieba
import datasetPre as dataPre
import os
from gensim import corpora, models
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim as ldagensim
from sklearn import svm
import re
import numpy as np
from collections import Counter
import copy

class LDA(dataPre.DatasetPre):
    def __init__(self):
        super(LDA,self).__init__()
        self.path = './trainset/'
        pass

    def trainData(self):
        # f = open(self.path + 'inf.txt')
        # f_names = f.read().split(',')
        # print(f_names)
        # f.close()
        '''
            cite:https://zhuanlan.zhihu.com/p/106980996
        '''
        #用于提取训练数据，仅一次执行
        if len(os.listdir(self.path))==1:
            self.textAbstract()
        if len(os.listdir('./testset')) == 1:
            self.testsetObt()
        txt,test_txt,dictionary,corpus=self.ladPrior()
        if len(os.listdir('./modelResult'))==0:
            self.ladModel(txt,dictionary,corpus)
        # self.superParamjustf(txt,dictionary,corpus)   #用于选择超参-主题数
        # self.visualPlay(56,txt,dictionary,corpus)    #用于综合评价lda模型,测试是无法运算后续代码，因为会一直中断，需要注释后运行后续代码
        self.svmClassifer(56,txt,test_txt,dictionary,corpus)

    def superParamjustf(self,txt,dictionary,corpus):
        x_list=[]
        y1_list=[]
        y2_list = []
        for i in range(5,60):
            try:
                temp = './modelResult/{}.model'.format('lda_{}_{}'.format(i, len(txt)))
                ldamodel=models.ldamodel.LdaModel.load(temp)
                perplexity=ldamodel.log_perplexity(corpus)
                cv_tmp=models.coherencemodel.CoherenceModel(model=ldamodel,texts=txt,dictionary=dictionary,coherence='c_v')
                x_list.append(i)
                y1_list.append(perplexity)
                y2_list.append(cv_tmp.get_coherence())
            except Exception as error:
                print(error)
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('topic num')
        ax1.set_ylabel('perplexity', color=color)
        ax1.plot(x_list, y1_list, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('coherence', color=color)
        ax2.plot(x_list, y2_list, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.show()

    def visualPlay(self,topic_num,txt,dictionary,corpus):
        temp = './modelResult/{}.model'.format('lda_{}_{}'.format(topic_num, len(txt)))
        tempmodel = models.ldamodel.LdaModel.load(temp)
        vis_data = ldagensim.prepare(tempmodel, corpus, dictionary)
        pyLDAvis.show(vis_data, open_browser=False)

    def svmClassifer(self,topic_num,txt,test_txt,dictionary,corpus):
        temp = './modelResult/{}.model'.format('lda_{}_{}'.format(topic_num, len(txt)))
        ldamodel = models.ldamodel.LdaModel.load(temp)
        topic_distribution = ldamodel.get_document_topics(corpus)
        inputx = np.zeros((len(txt), topic_num))
        label = np.zeros((len(txt), 1))
        self.svmDatapre(topic_distribution,inputx,label)
        classifier = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')
        classifier.fit(inputx, label)
        accuracy=sum(classifier.predict(inputx).reshape(-1,1)==label) / len(label)
        print('训练集SVM模型准确率：',accuracy)
        test_corpus=[dictionary.doc2bow(tmp) for tmp in test_txt]
        test_inputx=np.zeros((len(test_txt), topic_num))
        test_label = np.zeros((len(test_txt), 1))
        test_topic_distribution = ldamodel.get_document_topics(test_corpus)
        test_inputx,test_label=self.svmDatapre(test_topic_distribution, test_inputx, test_label)
        test_accuracy=sum(classifier.predict(test_inputx).reshape(-1,1)==test_label) / len(test_label)
        print('测试集SVM模型准确率：',test_accuracy)
        pass

    def svmDatapre(self,topic_distribution,inputx,label):
        if(len(topic_distribution)==200):
            categlory = len(self.text * 2)
        else:
            categlory = 2
        for i in range(len(topic_distribution)):
            tmp_topic_distribution = topic_distribution[i]
            for j in range(len(tmp_topic_distribution)):
                inputx[i][tmp_topic_distribution[j][0]] = tmp_topic_distribution[j][1]
            label[i]=int(i/categlory)
        return inputx,label

    def ladPrior(self):
        '''
        :func: 构造送入LDA的输入
        :return: 送入LDA的输入
        '''
        # 准备词典和词袋
        txt=[]
        test_txt=[]
        for file in self.text:
            for count in range(20):
                with open(self.path + file+str(count+1) + '.txt', 'r', encoding='ANSI') as file_object:
                    txt.append(jieba.lcut(file_object.read()))
                file_object.close()
        for file in self.text:
            for count in range(2):
                with open('./testset/' + file+str(count+1) + '.txt', 'r', encoding='ANSI') as file_object:
                    test_txt.append(jieba.lcut(file_object.read()))
                file_object.close()
        dictionary=corpora.Dictionary(txt)
        corpus=[dictionary.doc2bow(tmp) for tmp in txt]

        #此段代码加入TF-IDF，可以去除
        tfidf=models.TfidfModel(corpus)
        corpus_tfidf=tfidf[corpus]
        return txt,test_txt,dictionary,corpus

    def ladModel(self,txt,dictionary,corpus):
        #确定超参-主题数的影响
        for i in range(5,60):
            print('开始主题数为{}的模型训练'.format(i))
            print('当前的topic个数：{}'.format(i))
            print('当前的数据量：{}'.format(len(txt)))
            temp='lda_{}_{}'.format(i,len(txt))
            model=gensim.models.LdaModel(corpus,num_topics=i,id2word=dictionary)
            file_path='./modelResult/{}.model'.format(temp)
            model.save(file_path)
            print('---------------------')


if __name__ == '__main__':
    lda=LDA()
    lda.trainData()