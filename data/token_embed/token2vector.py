# -*- coding: utf-8 -*-
# python2

import os
import pandas as pd
import numpy as np 
from multiprocessing import Pool as ThreadPool #多进程

from extract import c2vec_parse #解析分词

from extract import w2vec_parse #解析分词

from gensim.models import FastText #FastText库

import collections #词频统计库
import wordcloud   #词云展示库

from PIL import Image # 图像处理库



########################################################Code2Vec#####################################

#解析sqlang数据
def c2vec_parse_sqlang_str(data_list):
    result = [' '.join(c2vec_parse('sqlang',line)) for line in data_list]
    return result
#解析csharp数据
def c2vec_parse_csharp_str(data_list):
    result = [' '.join(c2vec_parse('csharp',line)) for line in data_list]
    return result

#解析javang数据
def c2vec_parse_javang_str(data_list):
    result = [' '.join(c2vec_parse('javang',line)) for line in data_list]
    return result

#解析python数据
def c2vec_parse_python_str(data_list):
    result = [' '.join(c2vec_parse('python',line)) for line in data_list]
    return result


#分布式解析数据
def c2vec_parse_function(lang_type,lang_data,split_num):

    if lang_type=='sqlang':

        print('解析之前有%d条数据'%len(lang_data))

        query_data = lang_data['orgin_query'].tolist()
        query_split_list = [query_data[i:i + split_num] for i in range(0,len(query_data), split_num)]
        pool = ThreadPool(10)
        cut_query_list = pool.map(c2vec_parse_sqlang_str, query_split_list)
        pool.close()
        pool.join()
        cut_query = []
        for p in cut_query_list:
            cut_query += p
        print('cut_query条数：%d'%len(cut_query))

        code_data = lang_data['orgin_code'].tolist()
        code_split_list = [code_data[i:i + split_num] for i in range(0,len(code_data), split_num)]
        pool = ThreadPool(10)
        cut_code_list = pool.map(c2vec_parse_sqlang_str, code_split_list)
        pool.close()
        pool.join()
        cut_code = []
        for p in cut_code_list:
            cut_code += p
        print('cut_code条数：%d'%len(cut_code))

        qcont_data = lang_data['qcont'].tolist()
        qcont_split_list = [qcont_data[i:i + split_num] for i in range(0,len(qcont_data), split_num)]
        pool = ThreadPool(10)
        cut_qcont_list = pool.map(c2vec_c2vec_parse_sqlang_str, qcont_split_list)
        pool.close()
        pool.join()
        cut_qcont = []
        for p in cut_qcont_list:
            cut_qcont += p
        print('cut_qcont条数：%d'%len(cut_qcont))

        ccont_data = lang_data['ccont'].tolist()
        ccont_split_list = [ccont_data[i:i + split_num] for i in range(0,len(ccont_data), split_num)]
        pool = ThreadPool(10)
        cut_ccont_list = pool.map(c2vec_parse_sqlang_str, ccont_split_list)
        pool.close()
        pool.join()
        cut_ccont = []
        for p in cut_ccont_list:
            cut_ccont += p
        print('cut_ccont条数：%d'%len(cut_ccont))

        size = lang_data.shape[1]

        # 构建集
        lang_data.insert(size+0,'cut_query',cut_query)
        lang_data.insert(size+1,'cut_code',cut_code)
        lang_data.insert(size+2,'cut_qcont',cut_qcont)
        lang_data.insert(size+3,'cut_ccont',cut_ccont)

        # 重定位索引
        columns=['ID','orgin_query','cut_query','qcont','cut_qcont','orgin_code','cut_code','ccont','cut_ccont']
        # 重定位
        lang_data = lang_data.reindex(columns=columns)
        # 丢失解析无效
        parse_csv = lang_data.dropna(axis=0)

        print('解析之后有%d条数据' % len(parse_csv))

        return parse_csv

    if lang_type=='csharp':

        print('解析之前有%d条数据' % len(lang_data))

        query_data = lang_data['orgin_query'].tolist()
        query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
        pool = ThreadPool(10)
        cut_query_list = pool.map(c2vec_parse_csharp_str, query_split_list)
        pool.close()
        pool.join()
        cut_query = []
        for p in cut_query_list:
            cut_query += p
        print('cut_query条数：%d'%len(cut_query))

        code_data = lang_data['orgin_code'].tolist()
        code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
        pool = ThreadPool(10)
        cut_code_list = pool.map(c2vec_parse_csharp_str, code_split_list)
        pool.close()
        pool.join()
        cut_code = []
        for p in cut_code_list:
            cut_code += p
        print('cut_code条数：%d'%len(cut_code))

        qcont_data = lang_data['qcont'].tolist()
        qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
        pool = ThreadPool(10)
        cut_qcont_list = pool.map(c2vec_parse_csharp_str, qcont_split_list)
        pool.close()
        pool.join()
        cut_qcont = []
        for p in cut_qcont_list:
            cut_qcont += p
        print('cut_qcont条数：%d'%len(cut_qcont))

        ccont_data = lang_data['ccont'].tolist()
        ccont_split_list = [ccont_data[i:i + split_num] for i in range(0, len(ccont_data), split_num)]
        pool = ThreadPool(10)
        cut_ccont_list = pool.map(c2vec_parse_csharp_str, ccont_split_list)
        pool.close()
        pool.join()
        cut_ccont = []
        for p in cut_ccont_list:
            cut_ccont += p
        print('cut_ccont条数：%d'%len(cut_ccont))

        size=lang_data.shape[1]

        # 构建集
        lang_data.insert(size + 0, 'cut_query', cut_query)
        lang_data.insert(size + 1, 'cut_code', cut_code)
        lang_data.insert(size + 2, 'cut_qcont', cut_qcont)
        lang_data.insert(size + 3, 'cut_ccont', cut_ccont)

        # 重定位索引
        columns=['ID','orgin_query','cut_query','qcont','cut_qcont','orgin_code','cut_code','ccont', 'cut_ccont']
        # 重定位
        lang_data = lang_data.reindex(columns=columns)
        # 丢失解析无效
        parse_csv = lang_data.dropna(axis=0)

        print('解析之后有%d条数据' % len(parse_csv))

        return parse_csv

    if lang_type=='javang':

        print('解析之前有%d条数据' % len(lang_data))

        query_data = lang_data['orgin_query'].tolist()
        query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
        pool = ThreadPool(10)
        cut_query_list = pool.map(c2vec_parse_javang_str, query_split_list)
        pool.close()
        pool.join()
        cut_query = []
        for p in cut_query_list:
            cut_query += p
        print('cut_query条数：%d'%len(cut_query))

        code_data = lang_data['orgin_code'].tolist()
        code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
        pool = ThreadPool(10)
        cut_code_list = pool.map(c2vec_parse_javang_str, code_split_list)
        pool.close()
        pool.join()
        cut_code = []
        for p in cut_code_list:
            cut_code += p
        print('cut_code条数：%d'%len(cut_code))

        qcont_data = lang_data['qcont'].tolist()
        qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
        pool = ThreadPool(10)
        cut_qcont_list = pool.map(c2vec_parse_javang_str, qcont_split_list)
        pool.close()
        pool.join()
        cut_qcont = []
        for p in cut_qcont_list:
            cut_qcont += p
        print('cut_qcont条数：%d'%len(cut_qcont))

        ccont_data = lang_data['ccont'].tolist()
        ccont_split_list = [ccont_data[i:i + split_num] for i in range(0, len(ccont_data), split_num)]
        pool = ThreadPool(10)
        cut_ccont_list = pool.map(c2vec_parse_javang_str, ccont_split_list)
        pool.close()
        pool.join()
        cut_ccont = []
        for p in cut_ccont_list:
            cut_ccont += p
        print('cut_ccont条数：%d'%len(cut_ccont))

        size=lang_data.shape[1]

        # 构建集
        lang_data.insert(size + 0, 'cut_query', cut_query)
        lang_data.insert(size + 1, 'cut_code', cut_code)
        lang_data.insert(size + 2, 'cut_qcont', cut_qcont)
        lang_data.insert(size + 3, 'cut_ccont', cut_ccont)

        # 重定位索引
        columns=['ID','orgin_query','cut_query','qcont','cut_qcont','orgin_code','cut_code','ccont', 'cut_ccont']
        # 重定位
        lang_data = lang_data.reindex(columns=columns)
        # 丢失解析无效
        parse_csv = lang_data.dropna(axis=0)

        print('解析之后有%d条数据' % len(parse_csv))

        return parse_csv

    if lang_type=='python':

        print('解析之前有%d条数据' % len(lang_data))

        query_data = lang_data['orgin_query'].tolist()
        query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
        pool = ThreadPool(10)
        cut_query_list = pool.map(c2vec_parse_python_str, query_split_list)
        pool.close()
        pool.join()
        cut_query = []
        for p in cut_query_list:
            cut_query += p
        print('cut_query条数：%d'%len(cut_query))

        code_data = lang_data['orgin_code'].tolist()
        code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
        pool = ThreadPool(10)
        cut_code_list = pool.map(c2vec_parse_python_str, code_split_list)
        pool.close()
        pool.join()
        cut_code = []
        for p in cut_code_list:
            cut_code += p
        print('cut_code条数：%d'%len(cut_code))

        qcont_data = lang_data['qcont'].tolist()
        qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
        pool = ThreadPool(10)
        cut_qcont_list = pool.map(c2vec_parse_python_str, qcont_split_list)
        pool.close()
        pool.join()
        cut_qcont = []
        for p in cut_qcont_list:
            cut_qcont += p
        print('cut_qcont条数：%d'%len(cut_qcont))

        ccont_data = lang_data['ccont'].tolist()
        ccont_split_list = [ccont_data[i:i + split_num] for i in range(0, len(ccont_data), split_num)]
        pool = ThreadPool(10)
        cut_ccont_list = pool.map(c2vec_parse_python_str, ccont_split_list)
        pool.close()
        pool.join()
        cut_ccont = []
        for p in cut_ccont_list:
            cut_ccont += p
        print('cut_ccont条数：%d'%len(cut_ccont))

        size=lang_data.shape[1]

        # 构建集
        lang_data.insert(size + 0, 'cut_query', cut_query)
        lang_data.insert(size + 1, 'cut_code', cut_code)
        lang_data.insert(size + 2, 'cut_qcont', cut_qcont)
        lang_data.insert(size + 3, 'cut_ccont', cut_ccont)

        # 重定位索引
        columns=['ID','orgin_query','cut_query','qcont','cut_qcont','orgin_code','cut_code','ccont', 'cut_ccont']
        # 重定位
        lang_data = lang_data.reindex(columns=columns)
        # 丢失解析无效
        parse_csv = lang_data.dropna(axis=0)

        print('解析之后有%d条数据' % len(parse_csv))

        return parse_csv

#---------判断爬虫的词向量集合是否覆盖-----------
def c2vec_judge_cover(parse_path,c2vec_path,csvda_type,lang_type):
    #要实验仿真的数据
    lang_data = pd.read_csv(parse_path+'%s_fuse_%s_parse.csv'%(csvda_type,lang_type), index_col=0, sep='|')
    print('%s类型的%s数据用于实验仿真的共计%d条数据'%(csvda_type,lang_type,len(lang_data)))
    #训练词向量的数据
    corpus_data = pd.read_csv(c2vec_path+'%s_corpus.csv'%lang_type,index_col=0, sep='|')
    print('%s类型的词向量训练数据共计%d条数据'%(lang_type,len(corpus_data)))

    #######################判断所有的词向量训练数据是否包含实验仿真数据######################
    #抽取实验仿真ID
    lang_idlis = set(lang_data['ID'].tolist())    #列表
    #抽取训练向量ID
    corpus_idlis = set(corpus_data['ID'].tolist()) #列表
    #判断集合中所有元素是否包含在指定集合；是True，否False
    judge = lang_idlis.issubset(corpus_idlis)
    if judge:
        print('--------%s的词向量训练数据全部覆盖%s类型的实验数据OK------'%(lang_type,csvda_type))
    else:
        print('--------%s的词向量数据没有全部覆盖%s类型的实验数据!!!---'%(lang_type,csvda_type))


#-----------每行语料的收集处理---------
def c2vec_data_token(c2vec_path,lang_type,split_num): #数据过大拆分为n
    # 读取预测数据
    lang_data = pd.read_csv(c2vec_path+'%s_corpus.csv'%lang_type, index_col=0, sep='|')
    print('%s词向量训练数据处理之前有%d条数据'%(lang_type,len(lang_data)))

    # sqlang:602405,csharp:642662,javang:699700,python:434597

    # 丢失无效数据
    lang_data = lang_data.dropna(axis=0)
    print('%s词向量训练数据丢弃噪音有%d条数据'%(lang_type, len(lang_data)))

    # sqlang:602405,csharp:642662,javang:699700,python:434597

    lang_data_part1=lang_data[:int(len(lang_data)*1/10)]
    parse_data1 = c2vec_parse_function(lang_type,lang_data_part1,split_num)
    print('第一部分数据处理完毕！')

    lang_data_part2=lang_data[int(len(lang_data)*1/10)+1:int(len(lang_data)*2/10)]
    parse_data2 = c2vec_parse_function(lang_type,lang_data_part2,split_num)
    print('第二部分数据处理完毕！')

    lang_data_part3=lang_data[int(len(lang_data)*2/10)+1:int(len(lang_data)*3/10)]
    parse_data3 = c2vec_parse_function(lang_type,lang_data_part3,split_num)
    print('第三部分数据处理完毕！')

    lang_data_part4 = lang_data[int(len(lang_data)*3/10)+1:int(len(lang_data)*4/10)]
    parse_data4 = c2vec_parse_function(lang_type, lang_data_part4, split_num)
    print('第四部分数据处理完毕！')

    lang_data_part5 = lang_data[int(len(lang_data)*4/10)+1:int(len(lang_data)*5/10)]
    parse_data5 = c2vec_parse_function(lang_type, lang_data_part5, split_num)
    print('第五部分数据处理完毕！')

    lang_data_part6 = lang_data[int(len(lang_data)*5/10)+1:int(len(lang_data)*6/10)]
    parse_data6 = c2vec_parse_function(lang_type, lang_data_part6, split_num)
    print('第六部分数据处理完毕！')

    lang_data_part7 = lang_data[int(len(lang_data)*6/10)+1:int(len(lang_data)*7/10)]
    parse_data7 = c2vec_parse_function(lang_type, lang_data_part7, split_num)
    print('第七部分数据处理完毕！')

    lang_data_part8 = lang_data[int(len(lang_data)*7/10)+1:int(len(lang_data)*8/10)]
    parse_data8 = c2vec_parse_function(lang_type, lang_data_part8, split_num)
    print('第八部分数据处理完毕！')

    lang_data_part9 = lang_data[int(len(lang_data)*8/10)+1:int(len(lang_data)*9/10)]
    parse_data9 = c2vec_parse_function(lang_type, lang_data_part9, split_num)
    print('第九部分数据处理完毕！')

    lang_data_part10 = lang_data[int(len(lang_data)*9/10)+1:]
    parse_data10= c2vec_parse_function(lang_type, lang_data_part10, split_num)
    print('第十部分数据处理完毕！')

    # 合并数据
    parse_data=pd.concat([parse_data1,parse_data2,parse_data3,parse_data4,parse_data5,parse_data6,
                          parse_data7,parse_data8,parse_data9,parse_data10],axis=0)

    # 重新0-索引
    parse_data = parse_data.reset_index(drop=True)

    # 保存数据
    parse_data.to_csv(c2vec_path+'%s_corpus_tokens.csv'%lang_type,sep='|') #保存行名
    print('%s词向量数据数据处理完毕！'%lang_type)


def code_vector(c2vec_path,lang_type,split_num,words_top):
    # 保存文件
    c2vec_data_token(c2vec_path,lang_type,split_num)

    # 加载文件
    parse_data = pd.read_csv(c2vec_path+'%s_corpus_tokens.csv'%lang_type, index_col=0, sep='|')
    print('%s解析的原始数据共有%d条数据'% (lang_type,len(parse_data)))

    # sqlang:602396,csharp:642653,javang:699691,python:434588

    # 去除无效数据
    parse_data = parse_data.dropna(axis=0)
    print('%s去冗之后的数据共有%d条数据'% (lang_type,len(parse_data)))

    # sqlang:602394,csharp:642649,javang:699685,python:434585

    # 提取query
    cut_querys = parse_data['cut_query'].str.split(' ').tolist()
    cut_querys = list(filter(lambda x: x!=['-100'],cut_querys))
    # 提取query的context
    cut_qconts = parse_data['cut_qcont'].str.split(' ').tolist()
    cut_qconts = list(filter(lambda x: x!=['-100'], cut_qconts))
    # 提取code
    cut_codes = parse_data['cut_code'].str.split(' ').tolist()
    cut_codes = list(filter(lambda x: x!=['-100'], cut_codes))
    # 提取code的context
    cut_cconts = parse_data['cut_ccont'].str.split(' ').tolist()
    cut_cconts = list(filter(lambda x: x!=['-100'], cut_cconts))

    # 语料重组
    corpora = cut_querys+ cut_codes + cut_qconts+ cut_cconts

    print('%s语料加载完毕开始训练过程！......'%lang_type)

    # 训练word2vec词向量
    #--------------------------训练word2vec词向量-------------------------------#
    model=FastText(corpora,size=300,min_count=2,window=10,iter=20)
    #--------------------------训练word2vec词向量-------------------------------#
    model.save("./Code2Vec/%s_code2vec.model"%lang_type)
    print('%s数据的词向量模型训练完毕并保存！'%lang_type)
    # 保存词向量 word2vec
    model.wv.save_word2vec_format('./Word2Vec/%s_word2vec.txt'%lang_type,binary=False)
    # linux命令
    eping_cmd = 'sed \'1d\' ./Code2Vec/%s_code2vec.txt >  ../pairwise/code_embeddings/%s_glove.txt'%(lang_type,lang_type)
    # 转化为glove词向量
    os.popen(eping_cmd).read()

    #构建词表用于统计分析
    object_list=[]
    for seq in corpora:
        for token in seq:
            object_list.append(token)
    print('%s的语料中词典大小为%d'%(lang_type,len(object_list)))

    # sqlang:180891654,csharp:238344608,javang:234480598,python:178602792

    # 做语料的词频统计
    token_counts =collections.Counter(object_list)
    #  对分词做词频统计
    token_counts_top=token_counts.most_common(words_top)
    #  获取前20个高频的词并打印
    print(token_counts_top)
    #  词频的展示
    img = np.array(Image.open('./Code2Vec/wordcloud.png'))  # 定义词频背景
    #  设置背景图、显示词数、字体最大值
    wc = wordcloud.WordCloud(mask=img,max_words=100,max_font_size=200,background_color='white')
    #  结构化词的词频
    wc.generate_from_frequencies(token_counts)
    wc.to_file('./Code2Vec/%s_freq.jpg'%lang_type)

    # 提取query
    all_querys = parse_data['cut_query'].values.astype(np.str)
    query_text = ' '.join(all_querys.tolist())
    # 提取query的context
    all_qconts = parse_data['cut_qcont'].values.astype(np.str)
    qcont_text = ' '.join(all_qconts.tolist())
    # 提取code
    all_codes = parse_data['cut_code'].values.astype(np.str)
    code_text = ' '.join(all_codes.tolist())
    # 提取code的context
    all_cconts= parse_data['cut_ccont'].values.astype(np.str)
    ccont_text = ' '.join(all_cconts.tolist())

    # 语料保存txt
    with open(c2vec_path+'%s_corpus.txt'%lang_type,'a+') as f:
        f.write(query_text+' ')
        f.write(qcont_text+' ')
        f.write(code_text+' ')
        f.write(ccont_text+' ')
    print('%s的语料数据保存完毕！'%lang_type)



########################################################Word2Vec#####################################

#解析sqlang数据
def w2vec_parse_sqlang_str(data_list):
    result = [' '.join(w2vec_parse('sqlang',line)) for line in data_list]
    return result
#解析csharp数据
def w2vec_parse_csharp_str(data_list):
    result = [' '.join(w2vec_parse('csharp',line)) for line in data_list]
    return result

#解析javang数据
def w2vec_parse_javang_str(data_list):
    result = [' '.join(w2vec_parse('javang',line)) for line in data_list]
    return result

#解析python数据
def w2vec_parse_python_str(data_list):
    result = [' '.join(w2vec_parse('python',line)) for line in data_list]
    return result


#分布式解析数据
def w2vec_parse_function(lang_type,lang_data,split_num):

    if lang_type=='sqlang':

        print('解析之前有%d条数据'%len(lang_data))

        query_data = lang_data['orgin_query'].tolist()
        query_split_list = [query_data[i:i + split_num] for i in range(0,len(query_data), split_num)]
        pool = ThreadPool(10)
        cut_query_list = pool.map(w2vec_parse_sqlang_str, query_split_list)
        pool.close()
        pool.join()
        cut_query = []
        for p in cut_query_list:
            cut_query += p
        print('cut_query条数：%d'%len(cut_query))

        code_data = lang_data['orgin_code'].tolist()
        code_split_list = [code_data[i:i + split_num] for i in range(0,len(code_data), split_num)]
        pool = ThreadPool(10)
        cut_code_list = pool.map(w2vec_parse_sqlang_str, code_split_list)
        pool.close()
        pool.join()
        cut_code = []
        for p in cut_code_list:
            cut_code += p
        print('cut_code条数：%d'%len(cut_code))

        qcont_data = lang_data['qcont'].tolist()
        qcont_split_list = [qcont_data[i:i + split_num] for i in range(0,len(qcont_data), split_num)]
        pool = ThreadPool(10)
        cut_qcont_list = pool.map(w2vec_parse_sqlang_str, qcont_split_list)
        pool.close()
        pool.join()
        cut_qcont = []
        for p in cut_qcont_list:
            cut_qcont += p
        print('cut_qcont条数：%d'%len(cut_qcont))

        ccont_data = lang_data['ccont'].tolist()
        ccont_split_list = [ccont_data[i:i + split_num] for i in range(0,len(ccont_data), split_num)]
        pool = ThreadPool(10)
        cut_ccont_list = pool.map(w2vec_parse_sqlang_str, ccont_split_list)
        pool.close()
        pool.join()
        cut_ccont = []
        for p in cut_ccont_list:
            cut_ccont += p
        print('cut_ccont条数：%d'%len(cut_ccont))

        size = lang_data.shape[1]

        # 构建集
        lang_data.insert(size+0,'cut_query',cut_query)
        lang_data.insert(size+1,'cut_code',cut_code)
        lang_data.insert(size+2,'cut_qcont',cut_qcont)
        lang_data.insert(size+3,'cut_ccont',cut_ccont)

        # 重定位索引
        columns=['ID','orgin_query','cut_query','qcont','cut_qcont','orgin_code','cut_code','ccont','cut_ccont']
        # 重定位
        lang_data = lang_data.reindex(columns=columns)
        # 丢失解析无效
        parse_csv = lang_data.dropna(axis=0)

        print('解析之后有%d条数据' % len(parse_csv))

        return parse_csv

    if lang_type=='csharp':

        print('解析之前有%d条数据' % len(lang_data))

        query_data = lang_data['orgin_query'].tolist()
        query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
        pool = ThreadPool(10)
        cut_query_list = pool.map(w2vec_parse_csharp_str, query_split_list)
        pool.close()
        pool.join()
        cut_query = []
        for p in cut_query_list:
            cut_query += p
        print('cut_query条数：%d'%len(cut_query))

        code_data = lang_data['orgin_code'].tolist()
        code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
        pool = ThreadPool(10)
        cut_code_list = pool.map(w2vec_parse_csharp_str, code_split_list)
        pool.close()
        pool.join()
        cut_code = []
        for p in cut_code_list:
            cut_code += p
        print('cut_code条数：%d'%len(cut_code))

        qcont_data = lang_data['qcont'].tolist()
        qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
        pool = ThreadPool(10)
        cut_qcont_list = pool.map(w2vec_parse_csharp_str, qcont_split_list)
        pool.close()
        pool.join()
        cut_qcont = []
        for p in cut_qcont_list:
            cut_qcont += p
        print('cut_qcont条数：%d'%len(cut_qcont))

        ccont_data = lang_data['ccont'].tolist()
        ccont_split_list = [ccont_data[i:i + split_num] for i in range(0, len(ccont_data), split_num)]
        pool = ThreadPool(10)
        cut_ccont_list = pool.map(w2vec_parse_csharp_str, ccont_split_list)
        pool.close()
        pool.join()
        cut_ccont = []
        for p in cut_ccont_list:
            cut_ccont += p
        print('cut_ccont条数：%d'%len(cut_ccont))

        size=lang_data.shape[1]

        # 构建集
        lang_data.insert(size + 0, 'cut_query', cut_query)
        lang_data.insert(size + 1, 'cut_code', cut_code)
        lang_data.insert(size + 2, 'cut_qcont', cut_qcont)
        lang_data.insert(size + 3, 'cut_ccont', cut_ccont)

        # 重定位索引
        columns=['ID','orgin_query','cut_query','qcont','cut_qcont','orgin_code','cut_code','ccont', 'cut_ccont']
        # 重定位
        lang_data = lang_data.reindex(columns=columns)
        # 丢失解析无效
        parse_csv = lang_data.dropna(axis=0)

        print('解析之后有%d条数据' % len(parse_csv))

        return parse_csv

    if lang_type=='javang':

        print('解析之前有%d条数据' % len(lang_data))

        query_data = lang_data['orgin_query'].tolist()
        query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
        pool = ThreadPool(10)
        cut_query_list = pool.map(w2vec_parse_javang_str, query_split_list)
        pool.close()
        pool.join()
        cut_query = []
        for p in cut_query_list:
            cut_query += p
        print('cut_query条数：%d'%len(cut_query))

        code_data = lang_data['orgin_code'].tolist()
        code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
        pool = ThreadPool(10)
        cut_code_list = pool.map(w2vec_parse_javang_str, code_split_list)
        pool.close()
        pool.join()
        cut_code = []
        for p in cut_code_list:
            cut_code += p
        print('cut_code条数：%d'%len(cut_code))

        qcont_data = lang_data['qcont'].tolist()
        qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
        pool = ThreadPool(10)
        cut_qcont_list = pool.map(w2vec_parse_javang_str, qcont_split_list)
        pool.close()
        pool.join()
        cut_qcont = []
        for p in cut_qcont_list:
            cut_qcont += p
        print('cut_qcont条数：%d'%len(cut_qcont))

        ccont_data = lang_data['ccont'].tolist()
        ccont_split_list = [ccont_data[i:i + split_num] for i in range(0, len(ccont_data), split_num)]
        pool = ThreadPool(10)
        cut_ccont_list = pool.map(w2vec_parse_javang_str, ccont_split_list)
        pool.close()
        pool.join()
        cut_ccont = []
        for p in cut_ccont_list:
            cut_ccont += p
        print('cut_ccont条数：%d'%len(cut_ccont))

        size=lang_data.shape[1]

        # 构建集
        lang_data.insert(size + 0, 'cut_query', cut_query)
        lang_data.insert(size + 1, 'cut_code', cut_code)
        lang_data.insert(size + 2, 'cut_qcont', cut_qcont)
        lang_data.insert(size + 3, 'cut_ccont', cut_ccont)

        # 重定位索引
        columns=['ID','orgin_query','cut_query','qcont','cut_qcont','orgin_code','cut_code','ccont', 'cut_ccont']
        # 重定位
        lang_data = lang_data.reindex(columns=columns)
        # 丢失解析无效
        parse_csv = lang_data.dropna(axis=0)

        print('解析之后有%d条数据' % len(parse_csv))

        return parse_csv

    if lang_type=='python':

        print('解析之前有%d条数据' % len(lang_data))

        query_data = lang_data['orgin_query'].tolist()
        query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
        pool = ThreadPool(10)
        cut_query_list = pool.map(w2vec_parse_python_str, query_split_list)
        pool.close()
        pool.join()
        cut_query = []
        for p in cut_query_list:
            cut_query += p
        print('cut_query条数：%d'%len(cut_query))

        code_data = lang_data['orgin_code'].tolist()
        code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
        pool = ThreadPool(10)
        cut_code_list = pool.map(w2vec_parse_python_str, code_split_list)
        pool.close()
        pool.join()
        cut_code = []
        for p in cut_code_list:
            cut_code += p
        print('cut_code条数：%d'%len(cut_code))

        qcont_data = lang_data['qcont'].tolist()
        qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
        pool = ThreadPool(10)
        cut_qcont_list = pool.map(w2vec_parse_python_str, qcont_split_list)
        pool.close()
        pool.join()
        cut_qcont = []
        for p in cut_qcont_list:
            cut_qcont += p
        print('cut_qcont条数：%d'%len(cut_qcont))

        ccont_data = lang_data['ccont'].tolist()
        ccont_split_list = [ccont_data[i:i + split_num] for i in range(0, len(ccont_data), split_num)]
        pool = ThreadPool(10)
        cut_ccont_list = pool.map(w2vec_parse_python_str, ccont_split_list)
        pool.close()
        pool.join()
        cut_ccont = []
        for p in cut_ccont_list:
            cut_ccont += p
        print('cut_ccont条数：%d'%len(cut_ccont))

        size=lang_data.shape[1]

        # 构建集
        lang_data.insert(size + 0, 'cut_query', cut_query)
        lang_data.insert(size + 1, 'cut_code', cut_code)
        lang_data.insert(size + 2, 'cut_qcont', cut_qcont)
        lang_data.insert(size + 3, 'cut_ccont', cut_ccont)

        # 重定位索引
        columns=['ID','orgin_query','cut_query','qcont','cut_qcont','orgin_code','cut_code','ccont', 'cut_ccont']
        # 重定位
        lang_data = lang_data.reindex(columns=columns)
        # 丢失解析无效
        parse_csv = lang_data.dropna(axis=0)

        print('解析之后有%d条数据' % len(parse_csv))

        return parse_csv


#---------判断爬虫的词向量集合是否覆盖-----------
def w2vec_judge_cover(parse_path,w2vec_path,csvda_type,lang_type):
    #要实验仿真的数据
    lang_data = pd.read_csv(parse_path+'%s_fuse_%s_parse.csv'%(csvda_type,lang_type), index_col=0, sep='|')
    print('%s类型的%s数据用于实验仿真的共计%d条数据'%(csvda_type,lang_type,len(lang_data)))
    #训练词向量的数据
    corpus_data = pd.read_csv(w2vec_path+'%s_corpus.csv'%lang_type,index_col=0, sep='|')
    print('%s类型的词向量训练数据共计%d条数据'%(lang_type,len(corpus_data)))

    #######################判断所有的词向量训练数据是否包含实验仿真数据######################
    #抽取实验仿真ID
    lang_idlis = set(lang_data['ID'].tolist())    #列表
    #抽取训练向量ID
    corpus_idlis = set(corpus_data['ID'].tolist()) #列表
    #判断集合中所有元素是否包含在指定集合；是True，否False
    judge = lang_idlis.issubset(corpus_idlis)
    if judge:
        print('--------%s的词向量训练数据全部覆盖%s类型的实验数据OK------'%(lang_type,csvda_type))
    else:
        print('--------%s的词向量数据没有全部覆盖%s类型的实验数据!!!---'%(lang_type,csvda_type))


#-----------每行语料的收集清洁与干净处理---------
def w2vec_data_token(w2vec_path,lang_type,split_num): #数据过大拆分为n
    # 读取预测数据
    lang_data = pd.read_csv(w2vec_path+'%s_corpus.csv'%lang_type, index_col=0, sep='|')
    print('%s词向量训练数据处理之前有%d条数据'%(lang_type,len(lang_data)))

    # sqlang:602405,csharp:642662,javang:699700,python:434597

    # 丢失无效数据
    lang_data = lang_data.dropna(axis=0)
    print('%s词向量训练数据丢弃噪音有%d条数据'%(lang_type, len(lang_data)))

    # sqlang:602405,csharp:642662,javang:699700,python:434597

    lang_data_part1=lang_data[:int(len(lang_data)*1/10)]
    parse_data1 = w2vec_parse_function(lang_type,lang_data_part1,split_num)
    print('第一部分数据处理完毕！')

    lang_data_part2=lang_data[int(len(lang_data)*1/10)+1:int(len(lang_data)*2/10)]
    parse_data2 = w2vec_parse_function(lang_type,lang_data_part2,split_num)
    print('第二部分数据处理完毕！')

    lang_data_part3=lang_data[int(len(lang_data)*2/10)+1:int(len(lang_data)*3/10)]
    parse_data3 = w2vec_parse_function(lang_type,lang_data_part3,split_num)
    print('第三部分数据处理完毕！')

    lang_data_part4 = lang_data[int(len(lang_data)*3/10)+1:int(len(lang_data)*4/10)]
    parse_data4 = w2vec_parse_function(lang_type, lang_data_part4, split_num)
    print('第四部分数据处理完毕！')

    lang_data_part5 = lang_data[int(len(lang_data)*4/10)+1:int(len(lang_data)*5/10)]
    parse_data5 = w2vec_parse_function(lang_type, lang_data_part5, split_num)
    print('第五部分数据处理完毕！')

    lang_data_part6 = lang_data[int(len(lang_data)*5/10)+1:int(len(lang_data)*6/10)]
    parse_data6 = w2vec_parse_function(lang_type, lang_data_part6, split_num)
    print('第六部分数据处理完毕！')

    lang_data_part7 = lang_data[int(len(lang_data)*6/10)+1:int(len(lang_data)*7/10)]
    parse_data7 = w2vec_parse_function(lang_type, lang_data_part7, split_num)
    print('第七部分数据处理完毕！')

    lang_data_part8 = lang_data[int(len(lang_data)*7/10)+1:int(len(lang_data)*8/10)]
    parse_data8 = w2vec_parse_function(lang_type, lang_data_part8, split_num)
    print('第八部分数据处理完毕！')

    lang_data_part9 = lang_data[int(len(lang_data)*8/10)+1:int(len(lang_data)*9/10)]
    parse_data9 = w2vec_parse_function(lang_type, lang_data_part9, split_num)
    print('第九部分数据处理完毕！')

    lang_data_part10 = lang_data[int(len(lang_data)*9/10)+1:]
    parse_data10= w2vec_parse_function(lang_type, lang_data_part10, split_num)
    print('第十部分数据处理完毕！')

    # 合并数据
    parse_data=pd.concat([parse_data1,parse_data2,parse_data3,parse_data4,parse_data5,parse_data6,
                          parse_data7,parse_data8,parse_data9,parse_data10],axis=0)

    # 重新0-索引
    parse_data = parse_data.reset_index(drop=True)

    # 保存数据
    parse_data.to_csv(w2vec_path+'%s_corpus_tokens.csv'%lang_type,sep='|') #保存行名
    print('%s词向量数据数据处理完毕！'%lang_type)


def word_vector(w2vec_path,lang_type,split_num,words_top):
    # 保存文件
    w2vec_data_token(w2vec_path,lang_type,split_num)

    # 加载文件
    parse_data = pd.read_csv(w2vec_path+'%s_corpus_tokens.csv'%lang_type, index_col=0, sep='|')
    print('%s解析的原始数据共有%d条数据'% (lang_type,len(parse_data)))

    # sqlang:602396,csharp:642653,javang:699691,python:434588

    # 去除无效数据
    parse_data = parse_data.dropna(axis=0)
    print('%s去冗之后的数据共有%d条数据'% (lang_type,len(parse_data)))

    # sqlang:602394,csharp:642649,javang:699685,python:434585

    # 提取query
    cut_querys = parse_data['cut_query'].str.split(' ').tolist()
    cut_querys = list(filter(lambda x: x!=['-100'],cut_querys))
    # 提取query的context
    cut_qconts = parse_data['cut_qcont'].str.split(' ').tolist()
    cut_qconts = list(filter(lambda x: x!=['-100'], cut_qconts))
    # 提取code
    cut_codes = parse_data['cut_code'].str.split(' ').tolist()
    cut_codes = list(filter(lambda x: x!=['-100'], cut_codes))
    # 提取code的context
    cut_cconts = parse_data['cut_ccont'].str.split(' ').tolist()
    cut_cconts = list(filter(lambda x: x!=['-100'], cut_cconts))

    # 语料重组
    corpora = cut_querys+ cut_codes + cut_qconts+ cut_cconts

    print('%s语料加载完毕开始训练过程！......'%lang_type)

    # 训练word2vec词向量
    #--------------------------训练word2vec词向量-------------------------------#
    model=FastText(corpora,size=300,min_count=2,window=10,iter=20)
    #--------------------------训练word2vec词向量-------------------------------#
    model.save("./Word2Vec/%s_word2vec.model"%lang_type)
    #print('%s数据的词向量模型训练完毕并保存！'%lang_type)
    # 保存词向量 word2vec
    model.wv.save_word2vec_format('./Word2Vec/%s_word2vec.txt'%lang_type,binary=False)
    # linux命令
    eping_cmd = 'sed \'1d\' ./Word2Vec/%s_word2vec.txt >  ../pairwise/word_embeddings/%s_glove.txt'%(lang_type,lang_type)
    # 转化为glove词向量
    os.popen(eping_cmd).read()

    #构建词表用于统计分析
    object_list=[]
    for seq in corpora:
        for token in seq:
            object_list.append(token)
    print('%s的语料中词典大小为%d'%(lang_type,len(object_list)))

    # Code2Vec sqlang:180891654,csharp:238344608,javang:234480598,python:178602792

    # Word2Vec sqlang:206166568,csharp:           ,javang:     ,python:

    # # 做语料的词频统计
    # token_counts =collections.Counter(object_list)
    # #  对分词做词频统计
    # token_counts_top=token_counts.most_common(words_top)
    # #  获取前20个高频的词并打印
    # print(token_counts_top)
    # #  词频的展示
    # img = np.array(Image.open('./Word2Vec/wordcloud.png'))  # 定义词频背景
    # #  设置背景图、显示词数、字体最大值
    # wc = wordcloud.WordCloud(mask=img,max_words=100,max_font_size=200,background_color='white')
    # #  结构化词的词频
    # wc.generate_from_frequencies(token_counts)
    # wc.to_file('./Word2Vec/%s_freq.jpg'%lang_type)

    # # 提取query
    # all_querys = parse_data['cut_query'].values.astype(np.str)
    # query_text = ' '.join(all_querys.tolist())
    # # 提取query的context
    # all_qconts = parse_data['cut_qcont'].values.astype(np.str)
    # qcont_text = ' '.join(all_qconts.tolist())
    # # 提取code
    # all_codes = parse_data['cut_code'].values.astype(np.str)
    # code_text = ' '.join(all_codes.tolist())
    # # 提取code的context
    # all_cconts= parse_data['cut_ccont'].values.astype(np.str)
    # ccont_text = ' '.join(all_cconts.tolist())
    #
    # # 语料保存txt
    # with open(w2vec_path+'%s_corpus.txt'%lang_type,'a+') as f:
    #     f.write(query_text+' ')
    #     f.write(qcont_text+' ')
    #     f.write(code_text+' ')
    #     f.write(ccont_text+' ')
    # print('%s的语料数据保存完毕！'%lang_type)


#存储目录
parse_path = '../parse_corpus/'
c2vec_path = './c2vec_corpus/'
w2vec_path = './w2vec_corpus/'

#参数配置
sqlang_type = 'sqlang'
csharp_type = 'csharp'
javang_type = 'javang'
python_type = 'python'

#数据类型
single_type  = 'single'
mutiple_type ='mutiple'

#分割参数
split_num = 1000
words_top = 100

if __name__ == '__main__':
    #################################Code2Vec#######################
    #sqlang
    # c2vec_judge_cover(parse_path,c2vec_path,single_type,sqlang_type)
    # c2vec_judge_cover(parse_path,c2vec_path,mutiple_type,sqlang_type)
    # code_vector(c2vec_path,sqlang_type,split_num,words_top)
    #
    # #csharp
    # c2vec_judge_cover(parse_path,c2vec_path,single_type,csharp_type)
    # c2vec_judge_cover(parse_path,c2vec_path,mutiple_type,csharp_type)
    # code_vector(c2vec_path,csharp_type,split_num,words_top)
    #
    # #javang
    # c2vec_judge_cover(parse_path,c2vec_path,single_type,javang_type)
    # c2vec_judge_cover(parse_path,c2vec_path,mutiple_type,javang_type)
    # code_vector(c2vec_path,javang_type,split_num,words_top)
    #
    # #python
    # c2vec_judge_cover(parse_path,c2vec_path,single_type,python_type)
    # c2vec_judge_cover(parse_path,c2vec_path,mutiple_type,python_type)
    # code_vector(c2vec_path,python_type,split_num,words_top)

    #################################Word2Vec#######################
    #sqlang
    # w2vec_judge_cover(parse_path,w2vec_path,single_type,sqlang_type)
    # w2vec_judge_cover(parse_path,w2vec_path,mutiple_type,sqlang_type)
    # word_vector(w2vec_path,sqlang_type,split_num,words_top)

    #csharp
    #w2vec_judge_cover(parse_path,w2vec_path,single_type,csharp_type)
    #w2vec_judge_cover(parse_path,w2vec_path,mutiple_type,csharp_type)
    #word_vector(w2vec_path,csharp_type,split_num,words_top)

    #javang
    w2vec_judge_cover(parse_path,w2vec_path,single_type,javang_type)
    w2vec_judge_cover(parse_path,w2vec_path,mutiple_type,javang_type)
    word_vector(w2vec_path,javang_type,split_num,words_top)

    #python
    # w2vec_judge_cover(parse_path,w2vec_path,single_type,python_type)
    # w2vec_judge_cover(parse_path,w2vec_path,mutiple_type,python_type)
    # word_vector(w2vec_path,python_type,split_num,words_top)

