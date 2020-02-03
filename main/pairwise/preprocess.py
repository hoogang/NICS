# -*- coding: utf-8 -*-

import numpy as np
import pickle


#-----------------------------------训练集处理成三元组模式---------------------------------

def gen_train_samples(trans_path,candi_type,data_type,lang_type):
    #处理训练集
    train_ipath=trans_path+'/pairwise/data/%s_%s_%s_parse_train.txt'%(candi_type,data_type,lang_type)
    train_opath=trans_path+'/pairwise/pair/%s_%s_%s_train_triplets.txt'%(candi_type,data_type,lang_type)
    with open (train_ipath, 'r',encoding='utf-8') as fin:
        with open(train_opath,'w',encoding='utf-8') as fout:
            #正样本
            pos_list = []
            count=0  #计数
            # 循环读取每一行
            while True:
                line=fin.readline()
                # 如果不存在跳出循环
                if not line:
                    break
                #元素拆分
                line_info=line.split('\t')
                c_id=line_info[2]
                id=int(c_id.split('_')[1])
                if id==0:
                    count += 1
                    pos_code=line_info[3]
                    pos_list.append(pos_code)
                else:
                    query=line_info[1]
                    neg_code=line_info[3]
                    #写入（正样本，查询，负样本）
                    fout.write('\t'.join([pos_list[count-1],query,neg_code]) + '\n')

#-----------------------------------收集所有数据的词汇--------------------------------

def gen_vocab(trans_path,embed_path,candi_type,data_type,lang_type):

    words = []
    # 训练、验证、测试
    data_sets = ['%s_%s_%s_parse_train.txt'%(candi_type,data_type,lang_type),'%s_%s_%s_parse_valid.txt'%(candi_type,data_type,lang_type),'%s_%s_%s_parse_test.txt'%(candi_type,data_type,lang_type)]
    for set_name in data_sets:
        fin_path =trans_path+ '/pairwise/data/%s'%set_name
        with open(fin_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_in =line.strip('\n').split('\t')
                query=line_in[1].split(' ')
                code =line_in[3].split(' ')
                for r1 in query:
                    if r1 not in words:
                        words.append(r1)
                for r2 in code:
                    if r2 not in words:
                        words.append(r2)
    # 预测
    pred_path =trans_path+'/pred_corpus/%s_parse_pred.txt'%lang_type
    with open(pred_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_in = line.strip('\n').split('\t')
            query = line_in[1].split(' ')
            code = line_in[3].split(' ')
            for r1 in query:
                if r1 not in words:
                    words.append(r1)
            for r2 in code:
                if r2 not in words:
                    words.append(r2)

    fout_path = trans_path+'/pairwise'+embed_path+'/%s_%s_%s_vocab.txt'%(candi_type,data_type,lang_type)
    with open(fout_path,'w',encoding='utf-8') as fout:
        for i, j in enumerate(words):
            fout.write('{}\t{}\n'.format(i, j))


#-----------------------------------根据词表生成对应的embedding--------------------------------

def data_transform(trans_path,embed_path,candi_type,data_type,lang_type,embedding_size):

    vocab_in= trans_path+'/pairwise'+embed_path+'/%s_%s_%s_vocab.txt'%(candi_type,data_type,lang_type)

    # add 2 words: <PAD> and <UNK>
    clean_vocab_out = trans_path+'/pairwise'+embed_path+'/%s_%s_%s_clean_vocab.txt'%(candi_type,data_type,lang_type)

    embedding_in = trans_path+'/pairwise'+embed_path+'/%s_glove.txt'%lang_type

    embedding_out = trans_path+'/pairwise'+embed_path+'/%s_%s_%s_embedding.pkl'%(candi_type, data_type, lang_type)

    words = []
    with open(vocab_in, 'r', encoding='utf-8') as f1:
        for line in f1:
            word = line.strip('\n').split('\t')[1]
            words.append(word)
    print('%s候选的%s类型的%s语言中的vocab.txt总共有%d个词'%(candi_type,data_type,lang_type,len(words)))

    embedding_dic = {}
    rng = np.random.RandomState(None)
    pad_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))
    embeddings = []
    # 加载pad和unk词
    clean_words = ['<pad>', '<unk>']
    embeddings.append(pad_embedding.reshape(-1).tolist())
    embeddings.append(unk_embedding.reshape(-1).tolist())
    print('pad和unk服从均匀分布的随机变量初始化......')

    # 打开词向量
    with open(embedding_in, 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                line_info = line.strip('\n').split(' ')
                word = line_info[0]
                embedding = [float(val) for val in line_info[1:]]
                embedding_dic[word] = embedding
                if word in words:
                    #词在词典中
                    clean_words.append(word)
                    embeddings.append(embedding)
            except:
                print('这行加载失败: %s'%line.strip())

    print('%s候选的%s类型的%s语言在glove词表加上pad和unk总共有%d个词'%(candi_type,data_type,lang_type,len(clean_words)))

    print('%s候选的%s类型的%s语言的embeddings总共有%d个词'%(candi_type,data_type,lang_type,len(embeddings)))

    print('{}候选的{}类型的{}语言的embeddings的维度为:{}'.format(candi_type,data_type,lang_type,np.shape(embeddings)))

    # 保存在词库的的词
    with open(clean_vocab_out, 'w', encoding='utf-8') as f:
        for i, j in enumerate(clean_words):
            f.write('{}\t{}\n'.format(i, j))

    # 保存embedding为pickle文件
    with open(embedding_out, 'wb') as f2:
        pickle.dump(embeddings, f2)


#--------参数配置----------
trans_path= '/data/hugang/DeveCode/LRCode/data'

#普通化向量路径 word_embeddings或code_embeddings
embed_path='/word_embeddings'

#embed_path='/code_embeddings'

#候选类型
single_type='single'
mutiple_type='mutiple'
#数据类型
fuse_type ='fuse'
soqc_type ='soqc'
#语言类型
sqlang_type='sqlang'
csharp_type='csharp'
javang_type='javang'
python_type='python'

# 向量维度
embedding_size=300

#--------参数配置----------


if __name__ == '__main__':
    # sqlang
    gen_train_samples(trans_path,single_type,fuse_type,sqlang_type)
    gen_vocab(trans_path,embed_path,single_type,fuse_type,sqlang_type)
    data_transform(trans_path,embed_path,single_type,fuse_type,sqlang_type,embedding_size)

    gen_train_samples(trans_path,mutiple_type,fuse_type,sqlang_type)
    gen_vocab(trans_path,embed_path,mutiple_type, fuse_type, sqlang_type)
    data_transform(trans_path,embed_path,mutiple_type,fuse_type,sqlang_type,embedding_size)

    # csharp
    #gen_train_samples(trans_path,single_type,fuse_type,csharp_type)
    # gen_vocab(trans_path,embed_path,single_type,fuse_type,csharp_type)
    # data_transform(trans_path,embed_path,single_type,fuse_type,csharp_type,embedding_size)

    #gen_train_samples(trans_path,mutiple_type, fuse_type,csharp_type)
    # gen_vocab(trans_path,mutiple_type, fuse_type,csharp_type)
    # data_transform(trans_path,mutiple_type, fuse_type,csharp_type,embedding_size)

    # javang
    #gen_train_samples(trans_path,single_type, fuse_type, javang_type)
    # gen_vocab(trans_path,embed_path,single_type, fuse_type, javang_type)
    # data_transform(trans_path,embed_path,single_type, fuse_type, javang_type, embedding_size)

    #gen_train_samples(trans_path,mutiple_type, fuse_type, javang_type)
    # gen_vocab(trans_path,embed_path,mutiple_type, fuse_type, javang_type)
    # data_transform(trans_path,embed_path,mutiple_type, fuse_type, javang_type, embedding_size)

    # python
    #gen_train_samples(trans_path,single_type, fuse_type, python_type)
    #gen_vocab(trans_path,embed_path,single_type, fuse_type, python_type)
    #data_transform(trans_path,embed_path,single_type, fuse_type, python_type, embedding_size)

    #gen_train_samples(mutiple_type, fuse_type, python_type)
    #gen_vocab(trans_path,embed_path,mutiple_type, fuse_type, python_type)
    #data_transform(trans_path,embed_path,mutiple_type, fuse_type, python_type, embedding_size)

    ###################################################################
    gen_train_samples(trans_path,mutiple_type, soqc_type, sqlang_type)
    gen_vocab(trans_path,embed_path,mutiple_type, soqc_type, sqlang_type)
    data_transform(trans_path,embed_path,mutiple_type, soqc_type, sqlang_type, embedding_size)

    #gen_train_samples(trans_path,mutiple_type, soqc_type, python_type)
    #gen_vocab(trans_path,embed_path,mutiple_type, soqc_type, python_type)
    #data_transform(trans_path,embed_path,mutiple_type, soqc_type, python_type, embedding_size)
