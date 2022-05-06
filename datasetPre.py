#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Xu
# @date 2022/5/5
# @file datasetPre.py
import jieba

class DatasetPre():
    def __init__(self):
        self.file='./fiction'
        self.text=["鹿鼎记","天龙八部","笑傲江湖","倚天屠龙记","射雕英雄传","书剑恩仇录", "神雕侠侣", "碧血剑", "飞狐外传", "侠客行"]
        self.rabbish=['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库','\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......', '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']

    def content_clean(self,data):
        for i in self.rabbish:
            data = data.replace(i, '')
        return data


    def textAbstract(self):
        path=self.file
        files = self.text
        with open('./trainset/stop_word.txt') as file:
            stop_word_list = file.read()
        file.close()
        data_txt = []
        for file in files:
            with open((path + '\\' + file + '.txt' ), 'r', encoding='ANSI') as f:
                data = f.read()
                data = self.content_clean(data)
                data = jieba.lcut(data)
                tmp=[]
                for word in data:
                    if word not in stop_word_list:
                        tmp.append(word)
                data_txt.append(tmp)
            f.close()

        for txt, file in zip(data_txt, files):
            self.random_spilt(txt, file)
        return data_txt, files


    def random_spilt(self,txt, file):
        num = len(txt)
        interval = int(num/20)
        for i in range(20):
            fragment = txt[interval*i:interval*i+500]
            with open(('.\\trainset' + '\\' + file + str(i+1) + '.txt'), 'w', encoding='ANSI') as f:
                for item in fragment:
                    f.write(item)

    def test_random_spilt(self,txt, file):
        num = len(txt)
        interval = int(num/2)
        for i in range(2):
            fragment = txt[interval*i:interval*i+500]
            with open(('.\\testset' + '\\' + file + str(i+1) + '.txt'), 'w', encoding='ANSI') as f:
                for item in fragment:
                    f.write(item)

    def testsetObt(self):
        path = self.file
        files = self.text
        with open('./testset/stop_word.txt') as file:
            stop_word_list = file.read()
        file.close()
        data_txt = []
        for file in files:
            with open((path + '\\' + file + '.txt'), 'r', encoding='ANSI') as f:
                data = f.read()
                data = self.content_clean(data)
                data = jieba.lcut(data)
                tmp = []
                for word in data:
                    if word not in stop_word_list:
                        tmp.append(word)
                data_txt.append(tmp)
            f.close()

        for txt, file in zip(data_txt, files):
            self.test_random_spilt(txt, file)

if __name__ == '__main__':
    datasetPre=DatasetPre()
