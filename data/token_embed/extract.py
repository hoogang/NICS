# -*- coding: utf-8 -*-
# python2

import re
import sys
sys.path.append("..")

import nltk
import inflection


#sqlang解析
from scripts.sqlang_structured import sqlang_code_parse
from scripts.sqlang_structured import sqlang_query_parse

#csharp解析
from scripts.csharp_structured import csharp_code_parse
from scripts.csharp_structured import csharp_query_parse

#javang解析
from scripts.javang_structured import javang_code_parse
from scripts.javang_structured import javang_query_parse

#python解析
from scripts.python_structured import python_code_parse
from scripts.python_structured import python_query_parse



###################################Code2Vec词向量##########################################
def c2vec_parse(lang_type,pro_Line):

    # 程序类型分词 '-100' 表示标签不存在
    if lang_type=='sqlang':
        # 数据存在
        if pro_Line and pro_Line!='-100':
            # 分拆单词a_b=a b
            #line=line.replace('_', ' ')

            # 匹配任意字符（包括换行符）
            p= r'\[c\]((?!\[c\]|\[/c\])[\s\S])*\[/c\]'
            # 抽取代码标签
            matcCode = re.finditer(p, pro_Line, re.I)
            # 代码标签 ['<code>string<code>']
            codeTag = [i.group() for i in matcCode]

            if codeTag !=[]:# 有代码标签
                #############纯代码处理##########
                codeStr = [i.lstrip('[c]').rstrip('[/c]') for i in codeTag]
                # 判断代码类型
                codeTok = [['-100'] if j == '-1000' else (sqlang_code_parse(j) if sqlang_code_parse(j) != '-1000'
                                                          else ['-100']) for j in codeStr]
                # 合并代码令牌
                cTokStr = [' '.join(tok) for tok in codeTok]
                #############纯代码处理##########

                ############纯文本处理############
                # 文本替换
                for (c,n) in zip(codeTag, range(len(codeTag))):
                    pro_Line=pro_Line.replace(c,'ctag%d'%(n+1))
                # 解析查询
                cutWords = sqlang_query_parse(pro_Line)
                # 分词拼接
                textCuter = ' '.join(cutWords)
                ############纯文本处理############

                # 文本代码重组
                for (c,n) in zip(cTokStr,range(len(cTokStr))):
                    textCuter=textCuter.replace('ctag%d'%(n+1),c)
                # 分词 保留符号和单纯数字
                tokens = textCuter.split(' ')

                return tokens

            else: #无代码标签
                #解析查询文本
                tokens = sqlang_query_parse(pro_Line)

                return tokens
        else:
            return ['-100']

    # 程序类型分词 '-100' 表示标签不存在
    if lang_type == 'csharp':
        # 数据存在
        if pro_Line and pro_Line!='-100':
            # 分拆单词a_b=a b
            # line=line.replace('_', ' ')

            # 匹配任意字符（包括换行符）
            p = r'\[c\]((?!\[c\]|\[/c\])[\s\S])*\[/c\]'
            # 抽取代码标签
            matcCode = re.finditer(p, pro_Line, re.I)
            # 代码标签 ['<code>string<code>']
            codeTag = [i.group() for i in matcCode]

            if codeTag != []: # 有代码标签
                #############纯代码处理##########
                codeStr = [i.lstrip('[c]').rstrip('[/c]') for i in codeTag]
                # 判断代码类型
                codeTok = [['-100'] if j == '-1000' else (csharp_code_parse(j) if csharp_code_parse(j) != '-1000'
                                                          else ['-100']) for j in codeStr]
                # 合并代码令牌
                cTokStr = [' '.join(tok) for tok in codeTok]
                #############纯代码处理##########

                ############纯文本处理############
                # # 文本替换
                for (c, n) in zip(codeTag, range(len(codeTag))):
                    pro_Line = pro_Line.replace(c, 'ctag%d' % (n + 1))
                # 解析查询
                cutWords = csharp_query_parse(pro_Line)
                # 分词拼接
                textCuter = ' '.join(cutWords)
                ############纯文本处理############

                # 文本代码重组
                for (c, n) in zip(cTokStr, range(len(cTokStr))):
                    textCuter = textCuter.replace('ctag%d' % (n + 1), c)
                # 分词 保留符号和单纯数字
                tokens = textCuter.split(' ')

                return tokens

            else:  # 无代码标签
                # 解析查询文本
                tokens = csharp_query_parse(pro_Line)

                return tokens
        else:
            return ['-100']

    # 程序类型分词 '-100' 表示标签不存在
    if lang_type == 'javang':
        # 数据存在
        if pro_Line and pro_Line!='-100':
            # 分拆单词a_b=a b
            # line=line.replace('_', ' ')

            # 匹配任意字符（包括换行符）
            p = r'\[c\]((?!\[c\]|\[/c\])[\s\S])*\[/c\]'
            # 抽取代码标签
            matcCode = re.finditer(p, pro_Line, re.I)
            # 代码标签 ['<code>string<code>']
            codeTag = [i.group() for i in matcCode]

            if codeTag != []:  # 有代码标签
                #############纯代码处理##########
                codeStr = [i.lstrip('[c]').rstrip('[/c]') for i in codeTag]
                # 判断代码类型
                codeTok = [['-100'] if j == '-1000' else (javang_code_parse(j) if javang_code_parse(j) != '-1000'
                                                          else ['-100']) for j in codeStr]
                # 合并代码令牌
                cTokStr = [' '.join(tok) for tok in codeTok]
                #############纯代码处理##########

                ############纯文本处理############
                # # 文本替换
                for (c, n) in zip(codeTag, range(len(codeTag))):
                    pro_Line = pro_Line.replace(c, 'ctag%d' % (n + 1))
                # 解析查询
                cutWords = javang_query_parse(pro_Line)
                # 分词拼接
                textCuter = ' '.join(cutWords)
                ############纯文本处理############

                # 文本代码重组
                for (c, n) in zip(cTokStr, range(len(cTokStr))):
                    textCuter = textCuter.replace('ctag%d' % (n + 1), c)
                # 分词 保留符号和单纯数字
                tokens = textCuter.split(' ')

                return tokens

            else:  # 无代码标签
                # 解析查询文本
                tokens = javang_query_parse(pro_Line)

                return tokens
        else:
            return ['-100']

    # 程序类型分词 '-100' 表示标签不存在
    if lang_type == 'python':
        # 数据存在
        if pro_Line and pro_Line!='-100':
            # 分拆单词a_b=a b
            # line=line.replace('_', ' ')

            # 匹配任意字符（包括换行符）
            p = r'\[c\]((?!\[c\]|\[/c\])[\s\S])*\[/c\]'
            # 抽取代码标签
            matcCode = re.finditer(p, pro_Line, re.I)
            # 代码标签 ['<code>string<code>']
            codeTag = [i.group() for i in matcCode]

            if codeTag != []:  # 有代码标签
                #############纯代码处理##########
                codeStr = [i.lstrip('[c]').rstrip('[/c]') for i in codeTag]
                # 判断代码类型
                codeTok = [['-100'] if j == '-1000' else (python_code_parse(j) if python_code_parse(j) != '-1000'
                                                          else ['-100']) for j in codeStr]
                # 合并代码令牌
                cTokStr = [' '.join(tok) for tok in codeTok]
                #############纯代码处理##########

                ############纯文本处理############
                # 文本替换
                for (c, n) in zip(codeTag, range(len(codeTag))):
                    pro_Line = pro_Line.replace(c, 'ctag%d' % (n + 1))
                # 解析查询
                cutWords = python_query_parse(pro_Line)
                # 分词拼接
                textCuter = ' '.join(cutWords)
                ############纯文本处理############

                # 文本代码重组
                for (c, n) in zip(cTokStr, range(len(cTokStr))):
                    textCuter = textCuter.replace('ctag%d' % (n + 1), c)
                # 分词 保留符号和单纯数字
                tokens = textCuter.split(' ')

                return tokens

            else:  # 无代码标签
                # 解析查询文本
                tokens = python_query_parse(pro_Line)

                return tokens
        else:
            return ['-100']

###################################Word2Vec词向量############################################
def code_cuter(line):
    line = line.strip()
    line = line.replace("\\n", "\n")
    typedCode = nltk.word_tokenize(line)
    # 骆驼命名转下划线
    typedCode = inflection.underscore(' '.join(typedCode)).split(' ')
    cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
    # 全部小写化
    token_list = [x.lower().encode('utf8') for x in cut_tokens]
    return  token_list

###################################Word2Vec词向量##########################################
def w2vec_parse(lang_type,pro_Line):

    # 程序类型分词 '-100' 表示标签不存在
    if lang_type=='sqlang':
        # 数据存在
        if pro_Line and pro_Line!='-100':
            # 分拆单词a_b=a b
            #line=line.replace('_', ' ')

            # 匹配任意字符（包括换行符）
            p= r'\[c\]((?!\[c\]|\[/c\])[\s\S])*\[/c\]'
            # 抽取代码标签
            matcCode = re.finditer(p, pro_Line, re.I)
            # 代码标签 ['<code>string<code>']
            codeTag = [i.group() for i in matcCode]

            if codeTag !=[]:# 有代码标签
                #############纯代码处理##########
                codeStr = [i.lstrip('[c]').rstrip('[/c]') for i in codeTag]
                # 判断代码类型
                codeTok = [['-100'] if j == '-1000' else (code_cuter(j) if sqlang_code_parse(j) != '-1000'
                                                          else ['-100']) for j in codeStr]
                # 合并代码令牌
                cTokStr = [' '.join(tok) for tok in codeTok]
                #############纯代码处理##########

                ############纯文本处理############
                # 文本替换
                for (c,n) in zip(codeTag, range(len(codeTag))):
                    pro_Line=pro_Line.replace(c,'ctag%d'%(n+1))
                # 解析查询
                cutWords = sqlang_query_parse(pro_Line)
                # 分词拼接
                textCuter = ' '.join(cutWords)
                ############纯文本处理############

                # 文本代码重组
                for (c,n) in zip(cTokStr,range(len(cTokStr))):
                    textCuter=textCuter.replace('ctag%d'%(n+1),c)
                # 分词 保留符号和单纯数字
                tokens = textCuter.split(' ')

                return tokens

            else: #无代码标签
                #解析查询文本
                tokens = sqlang_query_parse(pro_Line)

                return tokens
        else:
            return ['-100']

    # 程序类型分词 '-100' 表示标签不存在
    if lang_type == 'csharp':
        # 数据存在
        if pro_Line and pro_Line!='-100':
            # 分拆单词a_b=a b
            # line=line.replace('_', ' ')

            # 匹配任意字符（包括换行符）
            p = r'\[c\]((?!\[c\]|\[/c\])[\s\S])*\[/c\]'
            # 抽取代码标签
            matcCode = re.finditer(p, pro_Line, re.I)
            # 代码标签 ['<code>string<code>']
            codeTag = [i.group() for i in matcCode]

            if codeTag != []: # 有代码标签
                #############纯代码处理##########
                codeStr = [i.lstrip('[c]').rstrip('[/c]') for i in codeTag]
                # 判断代码类型
                codeTok = [['-100'] if j == '-1000' else (code_cuter(j)  if csharp_code_parse(j) != '-1000'
                                                          else ['-100']) for j in codeStr]
                # 合并代码令牌
                cTokStr = [' '.join(tok) for tok in codeTok]
                #############纯代码处理##########

                ############纯文本处理############
                # 文本替换
                for (c, n) in zip(codeTag, range(len(codeTag))):
                    pro_Line = pro_Line.replace(c, 'ctag%d' % (n + 1))
                # 解析查询
                cutWords = csharp_query_parse(pro_Line)
                # 分词拼接
                textCuter = ' '.join(cutWords)
                ############纯文本处理############

                # 文本代码重组
                for (c, n) in zip(cTokStr, range(len(cTokStr))):
                    textCuter = textCuter.replace('ctag%d' % (n + 1), c)
                # 分词 保留符号和单纯数字
                tokens = textCuter.split(' ')

                return tokens

            else:  # 无代码标签
                # 解析查询文本
                tokens = csharp_query_parse(pro_Line)

                return tokens
        else:
            return ['-100']

    # 程序类型分词 '-100' 表示标签不存在
    if lang_type == 'javang':
        # 数据存在
        if pro_Line and pro_Line!='-100':
            # 分拆单词a_b=a b
            # line=line.replace('_', ' ')

            # 匹配任意字符（包括换行符）
            p = r'\[c\]((?!\[c\]|\[/c\])[\s\S])*\[/c\]'
            # 抽取代码标签
            matcCode = re.finditer(p, pro_Line, re.I)
            # 代码标签 ['<code>string<code>']
            codeTag = [i.group() for i in matcCode]

            if codeTag != []:  # 有代码标签
                #############纯代码处理##########
                codeStr = [i.lstrip('[c]').rstrip('[/c]') for i in codeTag]
                # 判断代码类型
                codeTok = [['-100'] if j == '-1000' else (code_cuter(j)  if javang_code_parse(j) != '-1000'
                                                          else ['-100']) for j in codeStr]
                # 合并代码令牌
                cTokStr = [' '.join(tok) for tok in codeTok]
                #############纯代码处理##########

                ############纯文本处理############
                # 文本替换
                for (c, n) in zip(codeTag, range(len(codeTag))):
                    pro_Line = pro_Line.replace(c, 'ctag%d' % (n + 1))
                # 解析查询
                cutWords = javang_query_parse(pro_Line)
                # 分词拼接
                textCuter = ' '.join(cutWords)
                ############纯文本处理############

                # 文本代码重组
                for (c, n) in zip(cTokStr, range(len(cTokStr))):
                    textCuter = textCuter.replace('ctag%d' % (n + 1), c)
                # 分词 保留符号和单纯数字
                tokens = textCuter.split(' ')

                return tokens

            else:  # 无代码标签
                # 解析查询文本
                tokens = javang_query_parse(pro_Line)

                return tokens
        else:
            return ['-100']

    # 程序类型分词 '-100' 表示标签不存在
    if lang_type == 'python':
        # 数据存在
        if pro_Line and pro_Line!='-100':
            # 分拆单词a_b=a b
            # line=line.replace('_', ' ')

            # 匹配任意字符（包括换行符）
            p = r'\[c\]((?!\[c\]|\[/c\])[\s\S])*\[/c\]'
            # 抽取代码标签
            matcCode = re.finditer(p, pro_Line, re.I)
            # 代码标签 ['<code>string<code>']
            codeTag = [i.group() for i in matcCode]

            if codeTag != []:  # 有代码标签
                #############纯代码处理##########
                codeStr = [i.lstrip('[c]').rstrip('[/c]') for i in codeTag]
                # 判断代码类型
                codeTok = [['-100'] if j == '-1000' else (code_cuter(j)  if python_code_parse(j) != '-1000'
                                                          else ['-100']) for j in codeStr]
                # 合并代码令牌
                cTokStr = [' '.join(tok) for tok in codeTok]
                #############纯代码处理##########

                ############纯文本处理############
                # 文本替换
                for (c, n) in zip(codeTag, range(len(codeTag))):
                    pro_Line = pro_Line.replace(c, 'ctag%d' % (n + 1))
                # 解析查询
                cutWords = python_query_parse(pro_Line)
                # 分词拼接
                textCuter = ' '.join(cutWords)
                ############纯文本处理############

                # 文本代码重组
                for (c, n) in zip(cTokStr, range(len(cTokStr))):
                    textCuter = textCuter.replace('ctag%d' % (n + 1), c)
                # 分词 保留符号和单纯数字
                tokens = textCuter.split(' ')

                return tokens

            else:  # 无代码标签
                # 解析查询文本
                tokens = python_query_parse(pro_Line)

                return tokens
        else:
            return ['-100']



#程序分词测试
if __name__ == '__main__':
    print(w2vec_parse('python','[c]groups = []\nuniquekeys = []\nfor k, g in groupby(data, keyfunc):\n groups.append(list(g)) # Store group iterator as a list\n uniquekeys.append(k)[/c] [c]-1000[/c] [c]-1000[/c] [c]-1000[/c] [c]from itertools import groupby\nthings = [("animal", "bear"), ("animal", "duck"), ("plant", "cactus"), ("vehicle", "speed boat"), ("vehicle", "school bus")]\nfor key, group in groupby(things, lambda x: x[0]):\n for thing in group:\n print "A %s is a %s." % (thing[1], key)\n print " "[/c] [c]-1000[/c] [c]-1000[/c] [c]lambda x: x[0][/c] [c]-1000[/c] [c]-1000[/c] [c]-1000[/c] [c]for key, group in groupby(things, lambda x: x[0]):\n listOfThings = " and ".join([thing[1] for thing in group])\n print key + "s: " + listOfThings + "."[/c]'))

    print(w2vec_parse('sqlang','The canonical way is to use the built-in cursor iterator. \n [c]curs.execute("select * from people")\nfor row in curs:\n print row\n[/c] \n \n You can use [c]fetchall()[/c] to get all rows at once. \n [c]for row in curs.fetchall():\n print row\n[/c] \n It can be convenient to use this to create a Python list containing the values returned: \n [c]curs.execute("select first_name from people")\nnames = [row[0] for row in curs.fetchall()]\n[/c] \n This can be useful for smaller result sets, but can have bad side effects if the result set is large. \n \n You have to wait for the entire result set to be returned to\nyour client process. \n You may eat up a lot of memory in your client to hold\nthe built-up list. \n It may take a while for Python to construct and deconstruct the\nlist which you are going to immediately discard anyways. \n \n \n If you know there"s a single row being returned in the result set you can call [c]fetchone()[/c] to get the single row. \n [c]curs.execute("select max(x) from t")\nmaxValue = curs.fetchone()[0]\n[/c] \n \n Finally, you can loop over the result set fetching one row at a time. In general, there"s no particular advantage in doing this over using the iterator. \n [c]row = curs.fetchone()\nwhile row:\n print row\n row = curs.fetchone()\n[/c]'))









