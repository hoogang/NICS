# -*- coding: utf-8 -*-

import random
import pandas as pd
import numpy as np


def creat_mutiple_fuse(parse_path,pred_path,lang_type,train_ratio,save_path,pair_path):

    # 加载单选数据
    mutiple_data = pd.read_csv(parse_path +'mutiple_fuse_%s_parse.csv' %lang_type, index_col=0, sep='|', encoding='utf-8')
    print('mutiple类型的%s数据处理之前的长度:\t%d'%(lang_type,len(mutiple_data)))

    # 去空字符串处理
    mutiple_data.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

    # 去除无效数据
    null_data = mutiple_data[mutiple_data.isnull().T.any()]
    acce_idx = list(set(mutiple_data['ID']).difference(set(null_data['ID'])))
    mutiple_data = mutiple_data[mutiple_data['ID'].isin(acce_idx)]
    print('mutiple类型的%s数据去冗之后的长度:\t%d' % (lang_type, len(mutiple_data)))

    # 分组排序（质量顺序变了）rank  0-k
    mutiple_data = mutiple_data.groupby("ID").apply(lambda row: row.sort_values(by="code_rank"))
    mutiple_data = mutiple_data.reset_index(drop=True)

    # 索引列表的确定
    column_list = ['ID','orgin_query','parse_query', 'orgin_code','parse_code','code_rank']
    lang_data = mutiple_data.reindex(columns=column_list)

    ###################################处理预测集##################################################
    # 长度不满足。
    nlen_data = lang_data[lang_data['parse_code'].str.split(' ').map(len) < 10]
    acce_idx = list(set(lang_data['ID']).difference(set(nlen_data['ID'])))
    ctlen_data = lang_data[lang_data['ID'].isin(acce_idx)]

    # 分组排序（质量顺序变了）rank  0-k
    ctlen_data = ctlen_data.groupby("ID").apply(lambda row: row.sort_values(by="code_rank"))
    ctlen_data = ctlen_data.reset_index(drop=True)

    # 索引列表的确定
    column_list = ['ID', 'orgin_query', 'parse_query', 'orgin_code', 'parse_code', 'code_rank']
    ctlen_data = ctlen_data.reindex(columns=column_list)
    #################################前200个#####################################################
    cid_rank = ctlen_data['ID'].value_counts()
    pidx_data = pd.DataFrame({'ID':cid_rank[:200].index.tolist()})  #拿出索引
    pred_data = lang_data[lang_data['ID'].isin(pidx_data['ID'])]  # 完整数据
    #################################前200个######################################################

    # 预测集长度
    pred_length = len(pred_data)
    # 保存数据，重新0-索引
    pred_data = pred_data.reset_index(drop=True)
    pred_data.to_csv(pred_path + '%s_pred.csv' % lang_type, sep='|', encoding='utf-8')
    cid_count = pred_data['ID'].value_counts()
    cid_count.to_csv(pred_path + '%s_count.csv' % lang_type, sep='|', encoding='utf-8')


    ######################################保存TXT数据######################
    # 提取query
    query_pred = pred_data['parse_query'].tolist()
    # 提取code
    code_pred  = pred_data['parse_code'].tolist()
    # 候选ID的长度
    pred_idx = pred_data[pred_data['code_rank'] == 0]['ID'].tolist()
    # 按顺序统计频次
    cid_count = cid_rank.reindex(index=pred_idx)
    # 候选ID频次列表
    pred_list = cid_count.tolist()

    # 查询id
    qid_pred = [['Q%d' % i] * j for i, j in zip(range(pred_length), pred_list)]
    qid_pred = [i for j in qid_pred for i in j]
    # 代码id
    cid_pred = [['C%d_%d' % (i, k) for k in range(j)] for i, j in zip(range(pred_length),pred_list)]
    cid_pred = [i for j in cid_pred for i in j]
    # 代码标签
    label_pred = [[1] + [0] * (j-1) for j in pred_list]
    label_pred = [i for j in label_pred for i in j]
    # 构建预测集
    pred_data = pd.DataFrame({'q_id': qid_pred, 'query':query_pred,'c_id': cid_pred,'code':code_pred,'label':label_pred})
    pred_data = pred_data.reindex(columns=['q_id', 'query', 'c_id', 'code', 'label'])
    print('%s预测集的长度:\t%d' % (lang_type, len(pred_data)))

    pred_data.to_csv(pred_path+'%s_parse_pred.txt'%lang_type,index=False, header=False, sep='\t',encoding='utf-8')
    print('TXT预测数据集保存完毕！')
    ###################################处理预测集###################################################

    ################################处理训练、验证、测试##############################################
    column_list = ['ID', 'parse_query', 'parse_code', 'code_rank']
    lang_data = lang_data.reindex(columns=column_list)
    # ID洗牌
    acce_idx = list(set(lang_data['ID']).difference(set(pidx_data['ID'])))
    # 新数据
    lang_data = lang_data[lang_data['ID'].isin(acce_idx)]
    print('mutiple类型的%s数据滤掉之后的长度:\t%d'%(lang_type,len(lang_data)))

    # 现有ID
    lang_idx = lang_data[lang_data['code_rank'] == 0]['ID'].tolist()

    # 切割点
    train_index = int(train_ratio * len(lang_idx))
    valid_index = int((train_ratio + 0.5 * (1-train_ratio))*len(lang_idx))
    ################################处理训练、验证、测试##############################################

    print("#############################训练集#######################################")
    # 训练数据,占比80%
    lang_train=lang_data[lang_data['ID'].isin(lang_idx[:train_index])]
    train_length=len(lang_train)  #长度

    # 提取query
    query_train = lang_train['parse_query'].tolist()
    # 提取code
    code_train = lang_train['parse_code'].tolist()

    # 候选ID的长度
    train_idx=  lang_train[lang_train['code_rank'] == 0]['ID'].tolist()
    # 按顺序统计频次
    cid_count = lang_train['ID'].value_counts().reindex(index=train_idx)
    # 候选ID频次列表
    train_list = cid_count.tolist()

    # 查询id
    qid_train = [['Q%d' % i] * j for i, j in zip(range(train_length), train_list)]
    qid_train = [i for j in qid_train for i in j]
    # 代码id
    cid_train = [['C%d_%d' % (i, k) for k in range(j)] for i, j in zip(range(train_length), train_list)]
    cid_train = [i for j in cid_train for i in j]
    # 代码标签
    label_train= [[1] + [0] *(j-1) for j in train_list]
    label_train = [i for j in label_train for i in j]

    # 构建训练集
    train_data=pd.DataFrame({'q_id':qid_train,'query':query_train,'c_id':cid_train,'code':code_train,'label':label_train})
    train_data=train_data.reindex(columns=['q_id','query','c_id','code','label'])
    print('mutiple类型的%s训练集的长度:\t%d' % (lang_type, len(train_data)))

    # 样本比值
    lables= train_data['label'].values  #样本的标签
    beta=np.sum(lables)/len(lables)
    print('Y+/Y-----正样本/样本：',beta)   #0.1
    # 保存csv格式不保存行名
    train_data.to_csv(save_path+'mutiple_fuse_%s_parse_train.csv'%lang_type,index=0,sep='|',encoding='utf-8')
    print('CSV训练数据集保存完毕！')
    # 保存TXT格式
    train_data.to_csv(pair_path+'mutiple_fuse_%s_parse_train.txt'%lang_type,index=False,header=False,sep='\t',encoding='utf-8')
    print('TXT训练数据集保存完毕！')

    print("#############################验证集#######################################")
    # 验证数据,占比10%
    lang_valid = lang_data[lang_data['ID'].isin(lang_idx[train_index+1:valid_index])]
    valid_length = len(lang_valid)  #长度

    # 提取query
    query_valid = lang_valid['parse_query'].tolist()
    # 提取code
    code_valid = lang_valid['parse_code'].tolist()

    # 候选ID的长度
    valid_idx = lang_valid[lang_valid['code_rank'] == 0]['ID'].tolist()
    # 按顺序统计频次
    cid_count = lang_valid['ID'].value_counts().reindex(index=valid_idx)
    # 候选ID频次列表
    valid_list = cid_count.tolist()

    # 查询id
    qid_valid= [['Q%d' % i] * j for i, j in zip(range(valid_length), valid_list)]
    qid_valid = [i for j in qid_valid for i in j]
    # 代码id
    cid_valid = [['C%d_%d' % (i, k) for k in range(j)] for i, j in zip(range(valid_length), valid_list)]
    cid_valid = [i for j in cid_valid for i in j]
    # 代码标签
    label_valid = [[1] + [0] * (j-1) for j in valid_list]
    label_valid = [i for j in label_valid for i in j]

    # 构建训练集
    valid_data = pd.DataFrame({'q_id': qid_valid, 'query': query_valid, 'c_id': cid_valid, 'code': code_valid, 'label': label_valid})
    valid_data = valid_data.reindex(columns=['q_id', 'query', 'c_id', 'code', 'label'])
    print('mutiple类型的%s验证集的长度:\t%d' % (lang_type, len(valid_data)))

    # 样本比值
    lables = valid_data['label'].values  # 样本的标签
    beta = np.sum(lables) / len(lables)
    print('Y+/Y-----正样本/样本：', beta)  # 0.1
    # 保存csv格式不保存行名
    valid_data.to_csv(save_path +'mutiple_fuse_%s_parse_valid.csv'%lang_type,index=0, sep='|',encoding='utf-8')
    print('CSV训练验证集保存完毕！')
    # 保存TXT格式
    valid_data.to_csv(pair_path +'mutiple_fuse_%s_parse_valid.txt'%lang_type,index=False, header=False,sep='\t', encoding='utf-8')
    print('TXT训练验证集保存完毕！')

    print("#############################测试集#######################################")
    # 验证数据,占比10%
    lang_test = lang_data[lang_data['ID'].isin(lang_idx[valid_index + 1:])]
    test_length = len(lang_test)  # 长度

    # 提取query
    query_test = lang_test['parse_query'].tolist()
    # 提取code
    code_test = lang_test['parse_code'].tolist()

    # 候选ID的长度
    test_idx = lang_test[lang_test['code_rank'] == 0]['ID'].tolist()
    # 按顺序统计频次
    cid_count = lang_test['ID'].value_counts().reindex(index=test_idx)
    # 候选ID频次列表
    test_list = cid_count.tolist()

    # 查询id
    qid_test = [['Q%d'%i]*j for i,j in zip(range(test_length),test_list)]
    qid_test = [i for j in qid_test for i in j]
    # 代码id
    cid_test = [['C%d_%d' % (i, k) for k in range(j)] for i,j in zip(range(test_length),test_list)]
    cid_test = [i for j in cid_test for i in j]
    # 代码标签
    label_test = [[1] + [0] * (j-1) for j in test_list]
    label_test = [i for j in label_test for i in j]
    # 构建训练集
    test_data = pd.DataFrame({'q_id': qid_test, 'query': query_test, 'c_id': cid_test, 'code': code_test, 'label': label_test})
    test_data = test_data.reindex(columns=['q_id', 'query', 'c_id', 'code', 'label'])
    print('mutiple类型的%s测试集的长度:\t%d' % (lang_type, len(test_data)))

    # 样本比值
    lables = test_data['label'].values  # 样本的标签
    beta = np.sum(lables) / len(lables)
    print('Y+/Y-----正样本/样本：', beta)  # 0.1
    # 保存csv格式不保存行名
    test_data.to_csv(save_path + 'mutiple_fuse_%s_parse_test.csv' %lang_type, index=0, sep='|',encoding='utf-8')
    print('CSV测试数据集保存完毕！')
    # 保存TXT格式
    test_data.to_csv(pair_path + 'mutiple_fuse_%s_parse_test.txt' %lang_type, index=False, header=False,sep='\t', encoding='utf-8')
    print('TXT测试数据集保存完毕！')


def creat_mutiple_soqc(parse_path,pred_path,lang_type,train_ratio,save_path,pair_path):

    # 加载单选数据
    mutiple_data = pd.read_csv(parse_path +'mutiple_soqc_%s_parse.csv' %lang_type, index_col=0, sep='|', encoding='utf-8')
    print('mutiple类型的%s数据处理之前的长度:\t%d' % (lang_type, len(mutiple_data)))

    # 去空字符串处理
    mutiple_data.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

    # 去除无效数据
    null_data = mutiple_data[mutiple_data.isnull().T.any()]
    acce_idx = list(set(mutiple_data['ID']).difference(set(null_data['ID'])))
    mutiple_data = mutiple_data[mutiple_data['ID'].isin(acce_idx)]
    print('mutiple类型的%s数据去冗之后的长度:\t%d' % (lang_type, len(mutiple_data)))

    # 分组排序（质量顺序变了）rank  0-k
    mutiple_data = mutiple_data.groupby("ID").apply(lambda row: row.sort_values(by="code_rank"))
    mutiple_data = mutiple_data.reset_index(drop=True)

    # 索引列表的确定
    column_list = ['ID','orgin_query','parse_query', 'orgin_code','parse_code','code_rank']
    lang_data = mutiple_data.reindex(columns=column_list)

    ###################################滤掉预测集##################################################
    pred_data = pd.read_csv(pred_path+'%s_pred.csv'%lang_type,index_col=0,sep='|',encoding='utf-8')
    # ID洗牌
    acce_idx = list(set(lang_data['ID']).difference(set(pred_data['ID'])))
    # 新数据
    lang_data = lang_data[lang_data['ID'].isin(acce_idx)]
    ###################################滤掉预测集###################################################

    ################################处理训练、验证、测试##############################################
    column_list = ['ID', 'parse_query', 'parse_code', 'code_rank']
    lang_data = lang_data.reindex(columns=column_list)

    # 现有ID
    lang_idx = lang_data[lang_data['code_rank'] == 0]['ID'].tolist()
    print('mutiple类型的%s数据截断之后的长度:\t%d' % (lang_type, len(lang_data)))

    # 切割点
    train_index = int(train_ratio * len(lang_idx))
    valid_index = int((train_ratio + 0.5 * (1-train_ratio))*len(lang_idx))
    ################################处理训练、验证、测试##############################################

    print("#############################训练集#######################################")
    # 训练数据,占比80%
    lang_train=lang_data[lang_data['ID'].isin(lang_idx[:train_index])]
    train_length=len(lang_train)  #长度

    # 提取query
    query_train = lang_train['parse_query'].tolist()
    # 提取code
    code_train = lang_train['parse_code'].tolist()

    # 候选ID的长度
    train_idx=  lang_train[lang_train['code_rank'] == 0]['ID'].tolist()
    # 按顺序统计频次
    cid_count = lang_train['ID'].value_counts().reindex(index=train_idx)
    # 候选ID频次列表
    train_list = cid_count.tolist()

    # 查询id
    qid_train = [['Q%d' % i] * j for i, j in zip(range(train_length), train_list)]
    qid_train = [i for j in qid_train for i in j]
    # 代码id
    cid_train = [['C%d_%d' % (i, k) for k in range(j)] for i, j in zip(range(train_length), train_list)]
    cid_train = [i for j in cid_train for i in j]
    # 代码标签
    label_train= [[1] + [0] *(j-1) for j in train_list]
    label_train = [i for j in label_train for i in j]

    # 构建训练集
    train_data=pd.DataFrame({'q_id':qid_train,'query':query_train,'c_id':cid_train,'code':code_train,'label':label_train})
    train_data=train_data.reindex(columns=['q_id','query','c_id','code','label'])
    print('mutiple类型的%s训练集的长度:\t%d' % (lang_type, len(train_data)))

    # 样本比值
    lables= train_data['label'].values  #样本的标签
    beta=np.sum(lables)/len(lables)
    print('Y+/Y-----正样本/样本：',beta)   #0.1
    # 保存csv格式不保存行名
    train_data.to_csv(save_path+'mutiple_soqc_%s_parse_train.csv'%lang_type,index=0,sep='|',encoding='utf-8')
    print('CSV训练数据集保存完毕！')
    # 保存TXT格式
    train_data.to_csv(pair_path+'mutiple_soqc_%s_parse_train.txt'%lang_type,index=False,header=False,sep='\t',encoding='utf-8')
    print('TXT训练数据集保存完毕！')

    print("#############################验证集#######################################")
    # 验证数据,占比10%
    lang_valid = lang_data[lang_data['ID'].isin(lang_idx[train_index+1:valid_index])]
    valid_length = len(lang_valid)  #长度

    # 提取query
    query_valid = lang_valid['parse_query'].tolist()
    # 提取code
    code_valid = lang_valid['parse_code'].tolist()

    # 候选ID的长度
    valid_idx = lang_valid[lang_valid['code_rank'] == 0]['ID'].tolist()
    # 按顺序统计频次
    cid_count = lang_valid['ID'].value_counts().reindex(index=valid_idx)
    # 候选ID频次列表
    valid_list = cid_count.tolist()

    # 查询id
    qid_valid= [['Q%d' % i] * j for i, j in zip(range(valid_length), valid_list)]
    qid_valid = [i for j in qid_valid for i in j]
    # 代码id
    cid_valid = [['C%d_%d' % (i, k) for k in range(j)] for i, j in zip(range(valid_length), valid_list)]
    cid_valid = [i for j in cid_valid for i in j]
    # 代码标签
    label_valid = [[1] + [0] * (j-1) for j in valid_list]
    label_valid = [i for j in label_valid for i in j]

    # 构建训练集
    valid_data = pd.DataFrame({'q_id': qid_valid, 'query': query_valid, 'c_id': cid_valid, 'code': code_valid, 'label': label_valid})
    valid_data = valid_data.reindex(columns=['q_id', 'query', 'c_id', 'code', 'label'])
    print('mutiple类型的%s验证集的长度:\t%d' % (lang_type, len(valid_data)))

    # 样本比值
    lables = valid_data['label'].values  # 样本的标签
    beta = np.sum(lables) / len(lables)
    print('Y+/Y-----正样本/样本：', beta)  # 0.1
    # 保存csv格式不保存行名
    valid_data.to_csv(save_path +'mutiple_soqc_%s_parse_valid.csv'%lang_type,index=0, sep='|',encoding='utf-8')
    print('CSV训练验证集保存完毕！')
    # 保存TXT格式
    valid_data.to_csv(pair_path +'mutiple_soqc_%s_parse_valid.txt'%lang_type,index=False, header=False,sep='\t', encoding='utf-8')
    print('TXT训练验证集保存完毕！')

    print("#############################测试集#######################################")
    # 测试数据,占比10%
    lang_test = lang_data[lang_data['ID'].isin(lang_idx[valid_index + 1:])]
    test_length = len(lang_test)  # 长度

    # 提取query
    query_test = lang_test['parse_query'].tolist()
    # 提取code
    code_test = lang_test['parse_code'].tolist()

    # 候选ID的长度
    test_idx = lang_test[lang_test['code_rank'] == 0]['ID'].tolist()
    # 按顺序统计频次
    cid_count = lang_test['ID'].value_counts().reindex(index=test_idx)
    # 候选ID频次列表
    test_list = cid_count.tolist()

    # 查询id
    qid_test = [['Q%d'%i]*j for i,j in zip(range(test_length),test_list)]
    qid_test = [i for j in qid_test for i in j]
    # 代码id
    cid_test = [['C%d_%d' % (i, k) for k in range(j)] for i,j in zip(range(test_length),test_list)]
    cid_test = [i for j in cid_test for i in j]
    # 代码标签
    label_test = [[1] + [0] * (j-1) for j in test_list]
    label_test = [i for j in label_test for i in j]
    # 构建训练集
    test_data = pd.DataFrame({'q_id': qid_test, 'query': query_test, 'c_id': cid_test, 'code': code_test, 'label': label_test})
    test_data = test_data.reindex(columns=['q_id', 'query', 'c_id', 'code', 'label'])
    print('mutiple类型的%s测试集的长度:\t%d' % (lang_type, len(test_data)))

    # 样本比值
    lables = test_data['label'].values  # 样本的标签
    beta = np.sum(lables) / len(lables)
    print('Y+/Y-----正样本/样本：', beta)  # 0.1
    # 保存csv格式不保存行名
    test_data.to_csv(save_path + 'mutiple_soqc_%s_parse_test.csv' %lang_type, index=0, sep='|',encoding='utf-8')
    print('CSV测试数据集保存完毕！')
    # 保存TXT格式
    test_data.to_csv(pair_path + 'mutiple_soqc_%s_parse_test.txt' %lang_type, index=False, header=False,sep='\t', encoding='utf-8')
    print('TXT测试数据集保存完毕！')


def creat_single_fuse(parse_path,pred_path,lang_type,train_ratio,save_path,pair_path):
    # 加载单选数据
    single_data=pd.read_csv(parse_path+'single_fuse_%s_parse.csv'%lang_type,index_col=0,sep='|',encoding='utf-8')
    print('single类型的%s数据处理之前的长度:\t%d'%(lang_type,len(single_data)))

    # 去空字符串处理
    single_data.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

    # 去除无效数据
    single_data = single_data.dropna(axis=0)
    single_data = single_data[single_data['parse_code'].str.split(' ').map(len) >= 20]
    print('single类型的%s数据去冗之后的长度:\t%d'%(lang_type,len(single_data)))

    # 索引列表的确定
    column_list = ['ID', 'orgin_query', 'parse_query', 'orgin_code', 'parse_code', 'code_rank']
    lang_data = single_data.reindex(columns=column_list)

    ###################################滤掉预测集##################################################
    # 滤候选最多的前200个
    pred_data=pd.read_csv(pred_path+'%s_pred.csv'%lang_type,index_col=0,sep='|',encoding='utf-8')
    # ID洗牌
    acce_idx = list(set(lang_data['ID']).difference(set(pred_data['ID'])))
    # 新数据
    lang_data = lang_data[lang_data['ID'].isin(acce_idx)]
    ###################################滤掉预测集###################################################

    ################################处理训练、验证、测试##############################################
    column_list = ['ID', 'parse_query', 'parse_code', 'code_rank']
    lang_data = lang_data.reindex(columns=column_list)

    # 现有ID
    lang_idx = lang_data[lang_data['code_rank'] == 0]['ID'].tolist()
    print('single类型的%s数据滤掉之后的长度:\t%d'%(lang_type, len(lang_data)))

    # 切割点
    train_index = int(train_ratio * len(lang_idx))
    valid_index = int((train_ratio + 0.5*(1-train_ratio)) * len(lang_idx))
    ################################处理训练、验证、测试##############################################

    print("#############################训练集#######################################")
    # 训练数据,占比80%
    lang_train = lang_data[lang_data['ID'].isin(lang_idx[:train_index])]
    train_length = len(lang_train)  # 长度

    # 提取query
    query_data=lang_train['parse_query'].tolist()
    query_train=[v for v in query_data for i in range(3)]
    # 提取code
    code_data=lang_train['parse_code'].tolist()
    code_train=['null']*train_length*3
    for i in range(0,train_length*3,3):
        code_train[i]=code_data[int(i/3)]
        random_list=random.sample([v for v in code_data if v!=code_data[int(i/3)]],3-1)
        for j in range(1,3):
            code_train[i+j]=random_list[j-1]
            #标签循环
    # 给定标签
    label_train = ['null']*train_length*3
    for i in range(0,train_length*3,3):
        label_train[i] = 1        #正样本
        for j in range(1,3):
            label_train[i+j] = 0  #负样本
    # 查询标签
    qid_train = ['Q%d'%i for i in range(train_length) for j in range(3)]
    # 代码标签
    cid_train = ['C%d_%d'%(i,j) for i in range(train_length) for j in range(3)]

    # 构建训练集
    train_data=pd.DataFrame({'q_id':qid_train,'query':query_train,'c_id':cid_train,'code':code_train,'label':label_train})
    train_data=train_data.reindex(columns=['q_id','query','c_id','code','label'])
    print('single类型的%s训练集的长度:\t%d' % (lang_type, len(train_data)))

    # 样本比值
    lables= train_data['label'].values  #样本的标签
    beta=np.sum(lables)/len(lables)
    print('Y+/Y-----正样本/样本：',beta)   #0.1
    # 保存csv格式不保存行名
    train_data.to_csv(save_path+'single_fuse_%s_parse_train.csv'%lang_type,index=0,sep='|',encoding='utf-8')
    print('CSV训练数据集保存完毕！')
    # 保存TXT格式
    train_data.to_csv(pair_path+'single_fuse_%s_parse_train.txt'%lang_type,index=False,header=False,sep='\t',encoding='utf-8')
    print('TXT训练数据集保存完毕！')

    print("#############################验证集#######################################")
    lang_valid = lang_data[lang_data['ID'].isin(lang_idx[train_index + 1:valid_index])]
    valid_length = len(lang_valid)  # 长度

    # 提取query
    query_data = lang_valid['parse_query'].tolist()
    query_valid = [v for v in query_data for i in range(3)]
    # 提取code
    code_data = lang_valid['parse_code'].tolist()
    code_valid = ['null'] * valid_length * 3
    for i in range(0, valid_length * 3, 3):
        code_valid[i] = code_data[int(i / 3)]
        random_list = random.sample([v for v in code_data if v != code_data[int(i/3)]], 3-1)
        for j in range(1,3):
            code_valid[i + j] = random_list[j - 1]
            # 标签循环
    # 给定标签
    label_valid = ['null'] * valid_length * 3
    for i in range(0, valid_length * 3, 3):
        label_valid[i] = 1  # 正样本
        for j in range(1, 3):
            label_valid[i+j] = 0  # 负样本
    # 查询标签
    qid_valid = ['Q%d' % i for i in range(valid_length) for j in range(3)]
    # 代码标签
    cid_valid= ['C%d_%d' % (i, j) for i in range(valid_length) for j in range(3)]

    # 构建训练集
    valid_data = pd.DataFrame({'q_id':qid_valid,'query':query_valid,'c_id':cid_valid,'code':code_valid,'label':label_valid})
    valid_data = valid_data.reindex(columns=['q_id','query','c_id','code','label'])
    print('single类型的%s训练集的长度:\t%d' % (lang_type, len(valid_data)))

    # 样本比值
    lables = valid_data['label'].values  # 样本的标签
    beta = np.sum(lables) / len(lables)
    print('Y+/Y-----正样本/样本：', beta)  # 0.1
    # 保存csv格式不保存行名
    valid_data.to_csv(save_path+'single_fuse_%s_parse_valid.csv'%lang_type,index=0,sep='|',encoding='utf-8')
    print('CSV训练数据集保存完毕！')
    # 保存TXT格式
    valid_data.to_csv(pair_path+'single_fuse_%s_parse_valid.txt'%lang_type,index=False,header=False,sep='\t',encoding='utf-8')
    print('TXT训练数据集保存完毕！')

    print("#############################测试集#######################################")
    # 测试数据,占比10%
    lang_test = lang_data[lang_data['ID'].isin(lang_idx[valid_index + 1:])]
    test_length = len(lang_test)  # 长度

    # 提取query
    query_data = lang_test['parse_query'].tolist()
    query_test  = [v for v in query_data for i in range(3)]
    # 提取code
    code_data = lang_test['parse_code'].tolist()
    code_test = ['null'] * test_length * 3
    for i in range(0, test_length * 3, 3):
        code_test[i] = code_data[int(i / 3)]
        random_list = random.sample([v for v in code_data if v != code_data[int(i/3)]], 3-1)
        for j in range(1,3):
            code_test[i + j] = random_list[j - 1]
            # 标签循环
    # 给定标签
    label_test = ['null'] * test_length * 3
    for i in range(0, test_length * 3, 3):
        label_test[i] = 1  # 正样本
        for j in range(1, 3):
            label_test[i+j] = 0  # 负样本
    # 查询标签
    qid_test = ['Q%d' % i for i in range(test_length) for j in range(3)]
    # 代码标签
    cid_test = ['C%d_%d' % (i, j) for i in range(test_length) for j in range(3)]

    # 构建训练集
    test_data = pd.DataFrame({'q_id': qid_test, 'query': query_test, 'c_id': cid_test, 'code': code_test, 'label': label_test})
    test_data = test_data.reindex(columns=['q_id', 'query', 'c_id', 'code', 'label'])
    print('single类型的%s训练集的长度:\t%d' % (lang_type, len(test_data)))

    # 样本比值
    lables = test_data['label'].values  # 样本的标签
    beta = np.sum(lables) / len(lables)
    print('Y+/Y-----正样本/样本：', beta)  # 0.1
    # 保存csv格式不保存行名
    test_data.to_csv(save_path + 'single_fuse_%s_parse_test.csv' % lang_type, index=0, sep='|', encoding='utf-8')
    print('CSV训练数据集保存完毕！')
    # 保存TXT格式
    test_data.to_csv(pair_path + 'single_fuse_%s_parse_test.txt' % lang_type, index=False, header=False, sep='\t',
                      encoding='utf-8')
    print('TXT训练数据集保存完毕！')


trans_path='/data/hugang/DeveCode/LRCode'

parse_path='../parse_corpus/'
pred_path='../pred_corpus/'
save_path='../prod_corpus/'
pair_path= trans_path+'/data'

sqlang_type='sqlang'  #93515条
csharp_type='csharp'  #68790条
javang_type='javang'  #68790条
python_type='python'  #68790条

train_ratio =0.8



if __name__ == '__main__':
    # fuse
    creat_mutiple_fuse(parse_path, pred_path,sqlang_type,train_ratio,save_path,pair_path)
    creat_mutiple_fuse(parse_path, pred_path,csharp_type,train_ratio,save_path,pair_path)
    creat_mutiple_fuse(parse_path, pred_path,javang_type,train_ratio,save_path,pair_path)
    creat_mutiple_fuse(parse_path, pred_path,python_type,train_ratio,save_path,pair_path)

    # fuse
    creat_single_fuse(parse_path,pred_path,sqlang_type,train_ratio,save_path,pair_path)
    creat_single_fuse(parse_path,pred_path,csharp_type,train_ratio,save_path,pair_path)
    creat_single_fuse(parse_path,pred_path,javang_type,train_ratio,save_path,pair_path)
    creat_single_fuse(parse_path,pred_path,python_type,train_ratio,save_path,pair_path)
    #
    # soqc
    creat_mutiple_soqc(parse_path,pred_path,sqlang_type,train_ratio,save_path,pair_path)
    creat_mutiple_soqc(parse_path,pred_path,python_type,train_ratio,save_path,pair_path)


