import numpy as np
import pandas as pd

from collections import Counter
from functools import reduce
from extract import parse

from gensim.summarization import bm25
from nltk.corpus import wordnet

#tf-idf的值自行计算
def tf(word, count):
    return count[word] / sum(count.values())

def n_containing(word, count_list):
    return np.sum([1 for count in count_list if word in count])

def idf(word, count_list):
    return np.log(len(count_list) / (1 + n_containing(word, count_list)))+1

def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)

#tsc论文的算法计算
def rank_code_score(pred_path,file_path,lang_type,pair_m,word_n):
    # 读取预测数据
    pred_data = pd.read_csv(pred_path+'%s_parse_pred.csv'%lang_type,index_col=0, sep='|', encoding='utf-8')
    # 数据的长度
    pred_length = len(pred_data)

    # 提取query
    all_querys = pred_data['orgin_query'].values.astype(np.str)
    # 提取code
    all_codes = pred_data['orgin_code'].values.astype(np.str)
    # 提取候选索引
    all_index = pred_data['index_list'].tolist()
    codes_ids = [i.split(' ') for i in all_index]
    # 提取software words
    parse_code = [parse(code) for code in all_codes]

    # 读取所有数据
    lang_data = pd.read_csv(file_path+'%s_parse.csv'%lang_type, index_col=0, sep='|', encoding='utf-8')
    #上下文信息长度
    lang_length= len(lang_data)
    # 提取q_contexts
    q_contexts = lang_data['qcont'].values.astype(np.str)
    # 提取q_contexts
    c_contexts = lang_data['ccont'].values.astype(np.str)
    # 提取dwtt
    dbwftf_scores = lang_data['all_score'].values
    # 合并query_str和code_str的context
    qc_context = np.char.add(np.char.add(q_contexts, ' '), c_contexts)
    # 提取context的software-words
    qc_pairs = [parse(context) for context in qc_context]
    # 用gensim建立BM25模型
    bm25_model1 = bm25.BM25(qc_pairs)
    # 计算平均逆文档频率
    pairs_idf = sum(map(lambda k: float(bm25_model1.idf[k]), bm25_model1.idf.keys())) / len(bm25_model1.idf.keys())

    # 返回代码的index
    rank10_list = []
    rank5_list = []
    rank1_list = []
    # 排名的位置
    true_rank = []
    #开始计算
    for id in range(pred_length):
        print('-------------------------代码的索引号------------------:', id)
        query_str = all_querys[id]
        #查询文本的分词
        query_words = parse(query_str)
        print('查询的原义词:\t', query_words)
        # 备选代码候选集
        codes_list = [int(i) for i in codes_ids[id]]
        #print('候选代码标记：\t', codes_list)
        # -------------------------------------------------------------------------------#
        #每条context进行打分
        lucene_scores = np.asarray(bm25_model1.get_scores(query_words,pairs_idf))
        #内容打分归一化
        content_scores = (lucene_scores-lucene_scores.min())/(lucene_scores.max()-lucene_scores.min())
        #质量打分归一化
        quality_scores = (dbwftf_scores-dbwftf_scores.min())/(dbwftf_scores.max()-dbwftf_scores.min())
        #打分值整合,加权
        pairs_scores = 0.95*content_scores+0.05*quality_scores
        # 创建 健-值对字典
        pairs_value = {}
        for i, v in zip(list(range(lang_length)), pairs_scores):
            pairs_value[i] = v
        # 将字典key-value按值排序，从大到小排序，最开始最大
        sorted_pairs_value = sorted(pairs_value.items(), key=lambda x: x[1], reverse=True)
        # 抽取前top-m的pairs打分
        topm_score = [x[1] for x in sorted_pairs_value[:pair_m]]
        print('Top-m的pairs的打分:\t', topm_score)
        topm_codes_value = [k_v for k_v in sorted_pairs_value if k_v[1] >= min(topm_score)]
        # 抽取前top-m的索引id
        topm_pairs_ids = [x[0] for x in topm_codes_value]
        print('Top-m的pairs的id:\t', topm_pairs_ids)
        #取出钱top-m的context
        rft_corpus = [qc_pairs[idx] for idx in topm_pairs_ids]
        #语料转成语句
        count_list = [Counter(i) for i in rft_corpus]
        #添加扩展词
        # -------------------------------------------------------------------------------
        expand_words = []
        for i, count in enumerate(count_list):
            #二范数处理，不作也行
            sums=np.sqrt(np.sum([np.square(tfidf(word, count, count_list)) for word in count]))
            #每个文档的tfidf
            words_tfidf  = {word: tfidf(word, count, count_list)/sums for word in count}
            #打分值从大到小排序
            sorted_tfidf = sorted(words_tfidf.items(), key=lambda d: d[1], reverse=True)
            #抽取前top-n打分的词
            topn_words = [index[0] for index in sorted_tfidf][:word_n]
            #词项累加
            expand_words += topn_words

        print('Bm25增加的扩展词:\t', expand_words)
        #-------------------------------------------------------------------------------#
        synet_words = []
        count = 0
        for x in query_words:
            for syn in wordnet.synsets(x):
                for l in syn.lemmas():
                    if (count < 3):
                        if l.name() not in synet_words:
                            synet_words.append(l.name())
                            count += 1
            count = 0
        print('Wnet增加的扩展词:\t',synet_words)
        #--------------------------------------------------------------------------------#
        #加入到query查询中
        expand_query = query_words + expand_words+synet_words
        # 按索引提取code
        index_code = [parse_code[i] for i in codes_list]
        # 用gensim建立BM25模型
        bm25_model2 = bm25.BM25(index_code)
        # 计算平均逆文档频率
        codes_idf = sum(map(lambda k: float(bm25_model2.idf[k]), bm25_model2.idf.keys())) / len(bm25_model2.idf.keys())
        #每条code进行打分
        codes_score = bm25_model2.get_scores(expand_query,codes_idf)
        # 创建键-值对字典
        codes_value = {}
        for i, v in zip(codes_list, codes_score):
            codes_value[i] = v
        # 将字典key-value按值排序，从大到小排序，最开始最大
        sorted_codes_value = sorted(codes_value.items(), key=lambda x: x[1], reverse=True)
        #print('代码ID对应的打分:\t', sorted_codes_value)
        #-----------------------------------------------------------------------------
        # 抽取前top-10的code打分
        top10_score = [x[1] for x in sorted_codes_value[:10]]
        print('Top-10的代码打分:', top10_score)
        top10_codes_value = [k_v for k_v in sorted_codes_value if k_v[1] >= min(top10_score)]
        # 抽取前top-10的索引id
        top10_ids = [x[0] for x in top10_codes_value]
        print('Top-10代码的ID:', top10_ids)
        # 排名在前10
        if id in top10_ids:
            mark10 = 1  # 真实标签存在索引里
        else:
            mark10 = 0  # 真实标签不在索引里
        rank10_list.append(mark10)

        # 抽取前top-5的code打分
        top5_score = [x[1] for x in sorted_codes_value[:5]]
        print('Top-5的代码打分:', top5_score)
        top5_codes_value = [k_v for k_v in sorted_codes_value if k_v[1] >= min(top5_score)]
        # 抽取前top-5的索引id
        top5_ids = [x[0] for x in top5_codes_value]
        print('Top-5代码的ID:', top5_ids)

        # 排名在前5
        if id in top5_ids:
            mark5 = 1  # 真实标签存在索引里
        else:
            mark5 = 0  # 真实标签不在索引里
        rank5_list.append(mark5)

        # 抽取前top-1的code打分
        top1_score = [x[1] for x in sorted_codes_value[:1]]
        print('Top-1的代码打分:', top1_score)
        top1_codes_value = [k_v for k_v in sorted_codes_value if k_v[1] >= min(top1_score)]
        # 抽取前top-1的索引id
        top1_ids = [x[0] for x in top1_codes_value]
        print('Top-1代码的ID:', top1_ids)

        # 排名在前1
        if id in top1_ids:
            mark1 = 1  # 真实标签存在索引里
        else:
            mark1 = 0  # 真实标签不在索引里
        rank1_list.append(mark1)

        # 打分最好的位置排名
        id_sorted = [x[0] for x in sorted_codes_value]
        print('代码ID的排名位置：\t', id_sorted)
        true_index = id_sorted.index(id) + 1
        true_rank.append(true_index)

    # 排名10，计算比例
    eval_10 = sum(rank10_list) / pred_length
    # 排名5，计算比例
    eval_5 = sum(rank5_list) / pred_length
    # 排名1，计算比例
    eval_1 = sum(rank1_list) / pred_length

    mrr_rate = sum([1 / x for x in true_rank]) / pred_length

    print('代码搜索recall@10的准确率：%f' % eval_10)
    print('代码搜索recall@5的准确率： %f' % eval_5)
    print('代码搜索recall@1的准确率： %f' % eval_1)
    print('代码搜索mrr_rate的准确率： %f' % mrr_rate)

    pred_data['rank10'] = pd.Series(rank10_list, index=pred_data.index)

    pred_data['rank5'] = pd.Series(rank5_list, index=pred_data.index)

    pred_data['rank1'] = pd.Series(rank1_list, index=pred_data.index)

    pred_data['true_rank'] = pd.Series(true_rank, index=pred_data.index)

    # 抽取属性分析关系
    column_key = ['orgin_query', 'parse_query', 'orgin_code', 'parse_code', 'index_list', 'true_rank', 'rank1',
                  'rank5', 'rank10']
    # 调整行名
    result_data = pred_data.reindex(columns=column_key)
    # 保存数据
    result_data.to_csv(pred_path + '%s_parse_pred_res5.csv' % lang_type, sep='|', encoding='utf-8')  # 保存行名
    print('预测数据处理完毕！')


# 参数配置
sqlang_type = 'sqlang'   # 93515条
csharp_type = 'csharp'   # 10548条


file_path = '../parse_corpus/'
pred_path = '../pred_corpus/'

pair_m=5
word_n=5

if __name__ == '__main__':
    rank_code_score(pred_path,file_path,sqlang_type,pair_m,word_n)
    rank_code_score(pred_path,file_path,csharp_type,pair_m,word_n)
