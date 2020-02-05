from gensim.models import KeyedVectors

import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def show_words(words_list,lang_type):  #词标签，词向量
    # 加载转换文件
    model = KeyedVectors.load_word2vec_format("./Word2Vec/%s_word2vec.txt"%lang_type, binary=False)
    # 输出词向量
    word_vectors = []
    for word in words_list:
        word_vectors.append(model.wv[word]) #加载词向量

    # TSE降维
    tsne = TSNE(n_components=2,perplexity=5,early_exaggeration=100,learning_rate=50,random_state=0,n_iter=10000,verbose=1)
    # 降维操作
    low_dim_embs = tsne.fit_transform(np.array(word_vectors))

    print(low_dim_embs.shape) # 转换后的词向量

    assert low_dim_embs.shape[0] == len(words_list)

    # 画图操作
    plt.figure(figsize=(6,6))
    # 画图
    for i, label in enumerate(words_list):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x,y), xytext=(2,4), textcoords='offset points', ha='right', va='bottom', fontsize=8)
    plt.savefig('%s_tsne.png'%lang_type)
    print('%s词向量可视化完毕！'%lang_type)


#-------------------------参数配置----------------------------------

sqlang_type = 'sqlang'   # 93515条
# 选的词语列表
sqlang_words=[
    'row','column','table','group','tagstr','tagint','codint','value','data','server','sql','database','if','with','from','for','to','on','in','into','between','or','and','but','where','what','when','which','order','by','of','as','like','set','select','insert','update','delete','join','inner','create','check','view','use','get','*','(',')',':','=']


csharp_type = 'csharp'   # 10548条
# 选的词语列表
csharp_words=[
    'or','and','if','else','use','public','static','get','class','void','string','return','var','private','int','type','list','false','true','method','system','number','str','convert','sender','select','add','length','write','read','row','column','like','change','call','need','item','create','find','look','tagstr','codstr','tagint','codint','{','}','<','>','(',')']

javang_type = 'javang'   # 10548条
# 选的词语列表
javang_words=['abstract','assert','boolean','break','byte','case','catch','char','class','const','continue','default','do','double','else','enum','final','finally','float','for','goto','if','import','int','interface','long','native','new','package','private','public','return','short','static','super','switch','this','try','void','while','tagstr','codstr','tagint','codint','{','}','<','>','(',')']


python_type = 'python'   # 10548条
# 选的词语列表
python_words=[
    'or','and','not','as','if','else','elif','str','int','float','list','tuple','dict','bool','set','false','none','true','assert', 'break', 'class', 'continue','del', 'for', 'range', 'in','global', 'from','import', 'is', 'lambda', 'nonlocal', 'pass', 'raise','def','return', 'try','except','finally', 'while', 'with', 'yield','tagstr','tagint','{','}','<','>','(',')']

#-------------------------参数配置----------------------------------

if __name__ == '__main__':
    show_words(sqlang_words,sqlang_type)
    #show_words(csharp_words,csharp_type)
    #show_words(javang_words,javang_type)
    #show_words(python_words,python_type)

