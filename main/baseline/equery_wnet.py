import numpy as np
import pandas as pd

from functools import reduce
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from extract import parse

#tf-idf的值自行计算
def tf(word, count):
    return count[word] / sum(count.values())

def n_containing(word, count_list):
    return np.sum([1 for count in count_list if word in count])

def idf(word, count_list):
    return np.log(len(count_list) / (1 + n_containing(word, count_list)))+1

def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def penn_to_wn(tag):
    if tag.startswith('N'):
        return 'n'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('R'):
        return 'r'
    return None


#tsc论文的算法计算
def rank_code_score(pred_path,lang_type):
    # 读取预测数据
    pred_data = pd.read_csv(pred_path+'%s_parse_pred.csv'%lang_type,index_col=0, sep='|', encoding='utf-8')
    # 数据的长度
    pred_length = len(pred_data)

    # 提取query
    all_querys = pred_data['orgin_query'].values.astype(np.str)
    # 提取code
    all_codes =  pred_data['orgin_code'].values.astype(np.str)
    # 提取候选索引
    all_index =  pred_data['index_list'].tolist()
    # 代码的索引   
    codes_ids =  [i.split(' ') for i in all_index]
    # 提取software words
    parse_code = [parse(code) for code in all_codes]

    #返回代码的index
    rank10_list = []
    rank5_list  = []
    rank1_list  = []
    #排名的位置
    true_rank = []
    #开始计算
    for id in range(pred_length):
        #print('-------------------------代码的索引号------------------:', id)
        query_str = all_querys[id]
        #查询文本的分词
        query_words = parse(query_str)
        #print('查询的原义词:\t', query_words)
        #备选代码候选集
        codes_list = [int(i) for i in codes_ids[id]]
        #print('候选代码标记：\t', codes_list)
        #-------------------------------------------------------------------------------#
        pos_words=pos_tag(query_words)

        equery_str=[]
        for word, word_tag in pos_words:
            tag = word_tag.lower()[0]
            if tag in ['a', 's', 'r', 'n', 'v']:
                synset = wordnet.synsets(word, pos=tag)
                if synset!=[]:
                    sim_words = [i.lemmas()[0].name() for i in synset]
                    #词干还原处理
                    lem_words = [wnl.lemmatize(w, tag) for w in sim_words]
                    synet_words=list(set(lem_words))
                    #更换同词性的近义词
                    change_query=[' '.join(query_words).replace(word,i) for i in synet_words]
                    equery_str.append(change_query)
        equery_str.append([' '.join(query_words)])
        #列表相加
        equery_list=reduce(lambda x,y:x+y,equery_str)
        get_phrases=[i.split(' ') for i in equery_list]
        #print('Wnet增加的短语:\t', get_phrases)
        #--------------------------------------------------------------------------------#
        #来自论文wordnet
        allkeys_num=[]
        for id in codes_list:
            code= parse_code[id]
            eachkey_num=sum([1 if len(set(code).intersection(set(query)))/max(len(code),len(query))>=1/250 else 0  for query in get_phrases])
            allkeys_num.append(eachkey_num)
        #计算代码得分
        codes_score=[(i/(sum(allkeys_num)+1))*10  for i in allkeys_num]
        #创建键-值对字典
        codes_value={}
        for i,v in zip(codes_list,codes_score):
            codes_value[i]=v
        #将字典key-value按值排序，从大到小排序，最开始最大
        sorted_codes_value =sorted(codes_value.items(), key=lambda x: x[1], reverse=True)
        #print('代码ID对应的打分:\t', sorted_codes_value)

        #抽取前top-10的code打分
        top10_score= [x[1] for x in sorted_codes_value[:10]]
        #print('Top-10的代码打分:', top10_score)
        top10_codes_value = [k_v for k_v in sorted_codes_value if k_v[1]>=min(top10_score)]
        #抽取前top-10的索引id
        top10_ids = [x[0] for x in top10_codes_value]
        #print('Top-10代码的ID:', top10_ids)

        # 排名在前10
        if id in top10_ids:
            mark10 = 1  # 真实标签存在索引里
        else:
            mark10 = 0  # 真实标签不在索引里
        rank10_list.append(mark10)

        # 抽取前top-5的code打分
        top5_score = [x[1] for x in sorted_codes_value[:5]]
        #print('Top-5的代码打分:', top5_score)
        top5_codes_value = [k_v for k_v in sorted_codes_value if k_v[1] >= min(top5_score)]
        # 抽取前top-5的索引id
        top5_ids = [x[0] for x in top5_codes_value]
        #print('Top-5代码的ID:', top5_ids)

        # 排名在前5
        if id in top5_ids:
            mark5 = 1  # 真实标签存在索引里
        else:
            mark5 = 0  # 真实标签不在索引里
        rank5_list.append(mark5)

        # 抽取前top-5的code打分
        top1_score = [x[1] for x in sorted_codes_value[:1]]
        #print('Top-1的代码打分:\t', top1_score)
        top1_codes_value = [k_v for k_v in sorted_codes_value if k_v[1] >= min(top1_score)]
        # 抽取前top-1的索引id
        top1_ids = [x[0] for x in top1_codes_value]
        #print('Top-1代码的ID:\t', top1_ids)

        # 排名在前1
        if id in top1_ids:
            mark1 = 1  # 真实标签存在索引里
        else:
            mark1 = 0  # 真实标签不在索引里
        rank1_list.append(mark1)

        #打分最好的位置排名
        id_sorted=[x[0] for x in sorted_codes_value]
        #print('代码ID的排名位置：\t',id_sorted)
        true_index = id_sorted.index(id)+1
        true_rank.append(true_index)

    #排名10，计算比例
    eval_10 = sum(rank10_list)/pred_length
    #排名5，计算比例
    eval_5  = sum(rank5_list)/pred_length
    #排名1，计算比例
    eval_1  = sum(rank1_list)/pred_length

    mrr_rate = sum([1/x for x in true_rank]) /pred_length

    print('代码搜索recall@10的准确率：%f'%eval_10)
    print('代码搜索recall@5的准确率： %f'%eval_5)
    print('代码搜索recall@1的准确率： %f'%eval_1)
    print('代码搜索mrr_rate的准确率： %f'%mrr_rate)

    pred_data['rank10']= pd.Series(rank10_list, index=pred_data.index)

    pred_data['rank5'] = pd.Series(rank5_list, index=pred_data.index)

    pred_data['rank1'] = pd.Series(rank1_list, index=pred_data.index)

    pred_data['true_rank']= pd.Series(true_rank, index=pred_data.index)

    # 抽取属性分析关系
    column_key = ['orgin_query', 'parse_query', 'orgin_code', 'parse_code', 'index_list', 'true_rank', 'rank1',
                  'rank5', 'rank10']
    # 调整行名
    result_data = pred_data.reindex(columns=column_key)
    # 保存数据
    result_data.to_csv(pred_path+'%s_parse_pred_res1.csv'%lang_type, sep='|', encoding='utf-8')  # 保存行名
    print('预测数据处理完毕！')



#参数配置
sqlang_type='sqlang'      #93515条
csharp_type='csharp'

pred_path = '../pred_corpus/'


if __name__ == '__main__':
    rank_code_score(pred_path,sqlang_type)
    rank_code_score(pred_path,csharp_type)
