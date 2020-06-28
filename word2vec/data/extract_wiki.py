#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@version :   python 3.7
@author  :   TobiasHu 
@time    :   2020/06/20 15:04
'''
import os
import json
import re
from string import punctuation
import pkuseg
seg = pkuseg.pkuseg()
dirname = os.path.abspath(os.path.dirname(__file__))
# 英文标点符号+中文标点符号
punc = punctuation + u'.,;《》？「」！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'
output = open('wiki_data.txt','w',encoding='utf-8')
count = 0
for filename in os.listdir(r'wiki_zh'):
    fpath = os.path.join(dirname+'/wiki_zh',filename)
    for files in os.listdir(fpath):
        filpath = os.path.join(fpath,files)
        with open(filpath,'r',encoding='utf-8') as content:
            for line in content:
                json_data = json.loads(line)["text"].split("\n")[1:]
                data = "".join(json_data)
                data = data.split('。')
                newdata = []
                for line in data:
                    line = re.sub(r"[{}]+".format(punc),"",line)
                    line = re.sub(r" ","",line)
                    wordlist = seg.cut(line)
                    count+=len(wordlist)
                    if count % 1000 == 0:
                        print("现在处理的词对数:",count)
                    newdata.append(" ".join(wordlist)) # 构造词与词之间使用空格隔开。
                newdata = "\n".join(newdata)
                output.write(newdata)
                if count >= 16000000: # 截获1600万个词对。
                    break
        if count >= 16000000:
            break
    if count >= 16000000:
        break
output.close()