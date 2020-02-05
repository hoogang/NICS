#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from bs4 import BeautifulSoup

#-----------------------------------------------解析XML文件的标签-----------------------------#
#滤除非法字符
def filter_inva_char(line):
    #去除非常用符号；防止解析有误
    line=re.sub('[^(0-9|a-z|A-Z|\-|#|/|%|_|,|\'|:|=|>|<|\"|;|-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+',' ', line)
    #中横线
    line=re.sub('-+','-',line)
    #下划线
    line=re.sub('_+','_',line)
    #去除横岗
    line=line.replace('|', ' ').replace('¦', ' ')
    #包括\r\t也清除了
    return line


#解析属性标签
def get_tag_param(rowTag,tag):
    try:
        output = rowTag[tag]
        return output
    except KeyError:
        return '-10000'

#解析code中数据
def parse_all_code(codeText):
    #  候选代码
    mucodes = []
    #  代码一个或多个标签
    for oneTag in codeText:
        code = oneTag.get_text().strip()
        #  代码片段10-1000字符,过滤字符长度
        if (len(code) >= 10 and len(code) <= 1000):
            #  过滤代码的关键片段
            if code[0] == '<' or code[0] == '=' or code[0] == '@' or code[0] == '$' or \
                    code[0:7].lower() == 'select' or code[0:7].lower() == 'update' or code[0:6].lower() == 'alter' or \
                    code[0:2].lower() == 'c:' or code[0:4].lower() == 'http' or code[0:4].lower() == 'hkey' or \
                    re.match(r'^[a-zA-Z0-9_]*$', code) is not None:
                #  加入代码
                mucodes.append('-1000')
            else:
                code = re.sub('\n+', '\n', re.sub(' +', ' ', filter_inva_char(code))).replace('\n', '\\n')
                #  满足要求
                mucodes.append(code)
        else:
            #  代码长度不满足要求
            mucodes.append('-1000')

    return mucodes

#判断code的标签
def judge_code_tag(codeTag):
    if len(codeTag) >= 1: #代码标签1或多个
        result=parse_all_code(codeTag)
        return result
    else: #无代码标签，可以用
        return '-100'

#-----------------------------------存储抽取的query对应的关键数据标签------------------------------
def process_qc_context(data_folder,lang_xml,answer_xml,split_num,change_xml,embed_folder,save_name):
    # 帖子id，接受id
    qidnum_lis,accept_lis=[],[]
    # 标题
    query_lis=[]
    # context
    qcont_lis=[]
    # 计数
    count1=0
    count2=0
    count3=0
    with open(data_folder+lang_xml,'r',encoding='utf-8') as f:
        for line in f:
            count1+=1
            if count1 %10000==0:
                print('查询阶段已经处理%d条数据'%count1)
            rowTag=BeautifulSoup(line,'lxml').row #row标签
            # 标题
            query=get_tag_param(rowTag,'title')
            if query!='-10000':
                accept=get_tag_param(rowTag,'acceptedanswerid')
                if accept!='-10000':
                    count2+=1
                    if count2 %10000==0:
                        print('查询阶段已经保存----%d条数据'%count2)
                    # 查询query # 分隔符'|'替换
                    query=filter_inva_char(query)
                    query_lis.append(query) #query一般没换行符
                    # 接受id
                    accept_lis.append(int(accept))
                    # 帖子id
                    qidnum = get_tag_param(rowTag, 'id')
                    qidnum_lis.append(int(qidnum))
                    # body属性
                    body = get_tag_param(rowTag, 'body')
                    soup = BeautifulSoup(body, 'lxml')
                    # 代码code
                    codeTag = soup.find_all('code') #code标签
                    # context
                    if judge_code_tag(codeTag) =='-100': #无代码标签,完全可用
                        count3 += 1
                        if count3 % 10000 == 0:
                            print('查询真实能用已经保存-------------%d条数据' % count3)
                        code_lf = re.compile(r'<code>')
                        nsoup = re.sub(code_lf, '[c]', str(soup))
                        code_ri = re.compile(r'</code>')
                        nsoup = re.sub(code_ri, '[/c]', str(nsoup))
                        content  = BeautifulSoup(nsoup, 'lxml')
                        qcont = content.findAll(text=True)
                        qcont = ' '.join([filter_inva_char(i) for i in qcont])
                        qcont = re.sub('\n+', '\n', re.sub(' +', ' ', qcont)).strip('\n').replace('\n', '\\n')
                        qcont = qcont if qcont else '-100'
                        qcont_lis.append(qcont)
                    else: #有一个或多个代码标签,且可解析
                        count3 += 1
                        if count3 % 10000 == 0:
                            print('查询真实能用已经保存-------------%d条数据' % count3)
                        mucodes = judge_code_tag(codeTag)
                        # 抽取无用id
                        nu_index = [id for id in range(len(mucodes)) if mucodes[id] == '-1000']
                        if nu_index!=[]:#存在不符合的代码
                            # 替换上下文
                            for i in nu_index:
                                codeText = [str(i) for i in codeTag]
                                nsoup = str(soup).replace(codeText[i],'<code>-1000</code>')
                                # 新的文本
                            code_lf = re.compile(r'<code>')
                            nsoup = re.sub(code_lf, '[c]', str(nsoup))
                            code_ri = re.compile(r'</code>')
                            nsoup = re.sub(code_ri, '[/c]', str(nsoup))
                            content = BeautifulSoup(nsoup, 'lxml')
                            qcont = content.findAll(text=True)
                            qcont = ' '.join([filter_inva_char(i) for i in qcont])
                            qcont = re.sub('\n+', '\n', re.sub(' +', ' ', qcont)).strip('\n').replace('\n', '\\n')
                            qcont_lis.append(qcont)
                        else:  #不存在不符合的代码
                            code_lf = re.compile(r'<code>')
                            nsoup = re.sub(code_lf, '[c]', str(soup))
                            code_ri = re.compile(r'</code>')
                            nsoup = re.sub(code_ri, '[/c]', str(nsoup))
                            content  = BeautifulSoup(nsoup, 'lxml')
                            qcont = content.findAll(text=True)
                            qcont = ' '.join([filter_inva_char(i) for i in qcont])
                            qcont = re.sub('\n+', '\n', re.sub(' +', ' ', qcont)).strip('\n').replace('\n', '\\n')
                            qcont_lis.append(qcont)

        print('总计查询了%d条数据'%count1)         #csharp:1089137,sqlang:1061859,javang:1100000,python:765848
        print('查询阶段总计保存了%d条数据'%count2)  #csharp:642662,sqlang:602405,javang:699700,python:434597
        print('查询最终能用的%d条上下文数据'%count3) #csharp:642662,sqlang:602405,javang:699700,python:434597

    print('###################################查询阶段执行完毕！###################################')

    # ###############################################找寻相关的XML文件#########################################
    print(len(qidnum_lis), len(query_lis), len(qcont_lis))

    # # 查询中索引的长度,#accept接受的标签
    index_num = len(accept_lis)
    print('查询中索引的长度:\t', index_num)  #csharp:642662；sqlang:602405,javang:699700,python:434597

    # 从小到大的序号排序
    sort_lis = sorted(accept_lis)
    # 来对索引index分割,正则匹配命令太长，必须切割！
    split_index = [sort_lis[i:i+split_num] for i in range(0,len(sort_lis),split_num)]

    #执行每个部分index
    parts_file=[]
    for i in range(len(split_index)):
        # linux执行命令
        eping_str = '|'.join(['(%s)'%str(s) for s in split_index[i]])
        # 每份xml命名
        epart_xml = change_xml.replace('.xml','Part%d.xml'%(i+1))
        # 添加到列表
        parts_file.append(data_folder+epart_xml)
        # 命令字符串
        eping_cmd = 'egrep  \'<row Id="(%s)"\' %s > %s'% (eping_str,data_folder+answer_xml,data_folder+epart_xml)
        # 执行命令符
        os.popen(eping_cmd).read() #得到part_xml文件
        print('索引的第%d部分检索完毕！'%(i+1))
    ping_cmd = 'cat %s > %s'%(' '.join(parts_file),data_folder+change_xml)
    # 执行命令
    os.popen(ping_cmd).read()   #得到change_xml文件
    print('XML整体文件合并完毕！')
    remove_cmd = 'rm %s' % data_folder+change_xml.replace('.xml','Part*.xml')
    os.popen(remove_cmd).read()
    print('XML部分文件删除完毕！') #csharp:642627,sqlang:602386,javang:697630,python:434591  部分id answer文件里无

    #-----------------------解析具有全部accept_lis索引的change_xml文件-----------------------------
    # 帖子id
    cidnum_lis=['-100']*index_num  #-100 没找到
    # 代码
    code_lis=['-100']*index_num   #-100 没找到
    # context
    ccont_lis=['-100']*index_num  #-100 没找到
    # 计数
    count1=0
    count2=0
    with open(data_folder+change_xml,'r',encoding='utf-8') as f:
        for line in f:  #总计  csharp:642627,sqlang:602386,javang:697630,python:434591
            count1+=1
            if count1 %10000==0:
                print('回复阶段已经处理%d条数据'%count1)
            sta= line.find("\"")
            end =line.find("\"", sta + 1)
            qid =int(line[(sta + 1):end])
            #  还是要判断
            if qid in accept_lis:
                # 并不一定从小到大排序且可能accept并没有完全遍历
                id = accept_lis.index(qid)
                # row标签
                rowTag=BeautifulSoup(line,'lxml').row
                # 帖子id
                cidnum=get_tag_param(rowTag,'id')
                cidnum_lis[id]=int(cidnum)
                # body属性
                body=get_tag_param(rowTag,'body')
                soup=BeautifulSoup(body,'lxml')
                # 代码code
                codeTag=soup.find_all('code')     #code标签
                mucodes=judge_code_tag(codeTag)   #code文本
                # 加标签
                code_str= ' '.join(['[c]%s[/c]'%code for code in mucodes]) if mucodes!='-100' else mucodes
                # 加入列表
                code_lis[id] = code_str
                # context
                if judge_code_tag(codeTag) == '-100':  #无代码标签,完全可用
                    count2 += 1
                    if count2 % 10000 == 0:
                        print('回复真实能用已经保存-------------%d条数据' % count2)
                    code_lf = re.compile(r'<code>')
                    nsoup = re.sub(code_lf, '[c]', str(soup))
                    code_ri = re.compile(r'</code>')
                    nsoup = re.sub(code_ri, '[/c]', str(nsoup))
                    content = BeautifulSoup(nsoup, 'lxml')
                    ccont = content.findAll(text=True)
                    ccont = ' '.join([filter_inva_char(i) for i in ccont])
                    ccont = re.sub('\n+', '\n', re.sub(' +', ' ', ccont)).strip('\n').replace('\n', '\\n')
                    ccont = ccont if ccont else '-100'
                    ccont_lis[id] = ccont
                else:  #有一个或多个代码标签,且可解析
                    count2 += 1
                    if count2 % 10000 == 0:
                        print('查询真实能用已经保存-------------%d条数据' % count2)
                    mucodes = judge_code_tag(codeTag)
                    # 抽取无用id
                    nu_index = [id for id in range(len(mucodes)) if mucodes[id] == '-1000']
                    if nu_index!= []: # 存在不符合的代码
                        # 替换上下文
                        for i in nu_index:
                            codeText = [str(i) for i in codeTag]
                            nsoup= str(soup).replace(codeText[i], '<code>-1000</code>')
                            # 新的文本
                        code_lf = re.compile(r'<code>')
                        nsoup = re.sub(code_lf, '[c]', str(nsoup))
                        code_ri = re.compile(r'</code>')
                        nsoup = re.sub(code_ri, '[/c]', str(nsoup))
                        content = BeautifulSoup(nsoup, 'lxml')
                        ccont = content.findAll(text=True)
                        ccont = ' '.join([filter_inva_char(i) for i in ccont])
                        ccont = re.sub('\n+', '\n', re.sub(' +', ' ', ccont)).strip('\n').replace('\n', '\\n')
                        ccont_lis[id]=ccont
                    else: # 不存在不符合的代码
                        code_lf = re.compile(r'<code>')
                        nsoup = re.sub(code_lf, '[c]', str(soup))
                        code_ri = re.compile(r'</code>')
                        nsoup = re.sub(code_ri, '[/c]', str(nsoup))
                        content = BeautifulSoup(nsoup, 'lxml')
                        ccont = content.findAll(text=True)
                        ccont = ' '.join([filter_inva_char(i) for i in ccont])
                        ccont = re.sub('\n+', '\n', re.sub(' +', ' ', ccont)).strip('\n').replace('\n', '\\n')
                        ccont_lis[id]=ccont

        print('回复阶段总计保存了%d条数据'%count1)   #csharp:642627,sqlang:602386,javang:697630,python:434591
        print('回复最终能用的%d条上下文数据'%count2) #csharp:642627,sqlang:602386,javang:697630,python:434591

    print('###################################回复阶段执行完毕！###################################')

    print(len(cidnum_lis), len(code_lis), len(ccont_lis))

    data_dict = {'ID':qidnum_lis,'acceptID':accept_lis,'orgin_query':query_lis,'qcont':qcont_lis,
                 'keyID':cidnum_lis,'orgin_code':code_lis,'ccont':ccont_lis}

    # 索引列表
    column_list = ['ID', 'acceptID', 'keyID', 'orgin_query', 'qcont', 'orgin_code', 'ccont']
    result_data = pd.DataFrame(data_dict,columns=column_list)
    # 最终大小
    print('最终的大小',result_data.shape) #格式 csharp:642662,sqlang:602405,javang:699700,python:434597

    # 单引号置换双引号 便于解析(主要是代码）
    for name in ['orgin_query','qcont','orgin_code','ccont']:
        result_data[name] = result_data[name].str.replace('\'', '\"')
    # 保存数据
    result_data.to_csv(embed_folder+save_name,encoding='utf-8',sep='|')


#参数的配置
data_folder='../data_crawl/corpus_xml/'
answer_xml='answer_all.xml'
split_num =10000

#XML标签数据
csharp_tag_xml='csharp_tag.xml'
csharp_ans_xml='csharp_ans.xml'
#CSV保存数据
csharp_save_name ='csharp_corpus.csv'

#XML标签数据
sqlang_tag_xml='sqlang_tag.xml'
sqlang_ans_xml='sqlang_ans.xml'
#CSV保存数据
sqlang_save_name='sqlang_corpus.csv'

#XML标签数据
javang_tag_xml='javang_tag.xml'
javang_ans_xml='javang_ans.xml'
#CSV保存数据
javang_save_name='javang_corpus.csv'

#XML标签数据
python_tag_xml='python_tag.xml'
python_ans_xml='python_ans.xml'
#CSV保存数据
python_save_name='python_corpus.csv'

embed_folder= './embed_corpus/'


if __name__ == '__main__':
    # csharp
    process_qc_context(data_folder,csharp_tag_xml,answer_xml,split_num,csharp_ans_xml,embed_folder,csharp_save_name)
    # sqlang
    process_qc_context(data_folder,sqlang_tag_xml,answer_xml,split_num,sqlang_ans_xml,embed_folder,sqlang_save_name)
    # javang
    process_qc_context(data_folder,javang_tag_xml,answer_xml,split_num,javang_ans_xml,embed_folder,javang_save_name)
    # python
    process_qc_context(data_folder,python_tag_xml,answer_xml,split_num,python_ans_xml,embed_folder,python_save_name)
