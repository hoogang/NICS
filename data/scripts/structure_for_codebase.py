#coding=utf-8
# python2

#多进程
from multiprocessing import Pool as ThreadPool
import pandas as pd

#sqlang解析
from  sqlang_structured import sqlang_code_parse
from  sqlang_structured import sqlang_query_parse

#csharp解析
from  csharp_structured import csharp_code_parse
from  csharp_structured import csharp_query_parse

#javang解析
from  javang_structured import javang_code_parse
from  javang_structured import javang_query_parse

#python解析
from  python_structured import python_code_parse
from  python_structured import python_query_parse


##########################################sqlang处理####################
def multi_query_prosqlang(data_list):
    result=[' '.join(sqlang_query_parse(line)) for line in data_list]
    return result

def multi_code_prosqlang(data_list):
    data_code=[line for line in data_list]
    result=[' '.join(sqlang_code_parse(line)) if sqlang_code_parse(line)!='-1000' else '-100' for line in data_code]
    return result

##########################################csharp处理####################
def multi_query_procsharp(data_list):
    result=[' '.join(csharp_query_parse(line)) for line in data_list]
    return result

def multi_code_procsharp(data_list):
    data_code=[line for line in data_list]
    result=[' '.join(csharp_code_parse(line)) if csharp_code_parse(line)!='-1000' else '-100' for line in data_code]
    return result

##########################################javang处理####################
def multi_query_projavang(data_list):
    result=[' '.join(javang_query_parse(line)) for line in data_list]
    return result

def multi_code_projavang(data_list):
    data_code=[line for line in data_list]
    result=[' '.join(javang_code_parse(line)) if javang_code_parse(line)!='-1000' else '-100' for line in data_code]
    return result

##########################################python处理####################
def multi_query_propython(data_list):
    result=[' '.join(python_query_parse(line)) for line in data_list]
    return result

def multi_code_propython(data_list):
    data_code=[line for line in data_list]
    result=[' '.join(python_code_parse(line)) if python_code_parse(line)!='-1000' else '-100' for line in data_code]
    return result

###########################################sqlang#############################################
def sqlang_query_code(sqlang_data,split_num):
    query_data=sqlang_data['orgin_query'].tolist()
    query_split_list=[query_data[i:i+split_num] for i in range(0,len(query_data),split_num)]
    pool= ThreadPool(10)
    query_list=pool.map(multi_query_prosqlang, query_split_list)
    pool.close()
    pool.join()
    query=[]
    for p in query_list:
        query+=p
    print('query条数：%d'%len(query))

    code_data=sqlang_data['orgin_code'].tolist()
    code_split_list=[code_data[i:i+split_num] for i in range(0,len(code_data),split_num)]
    pool= ThreadPool(10)
    code_list=pool.map(multi_code_prosqlang, code_split_list)
    pool.close()
    pool.join()
    code=[]
    for p in code_list:
        code+=p
    print('code条数：%d'%len(code))

    col_name = sqlang_data.columns.tolist()

    # 构建集
    sqlang_data.insert(col_name.index('orgin_query') + 1, 'parse_query', query)
    sqlang_data.insert(col_name.index('orgin_code') + 2, 'parse_code', code)

    return sqlang_data

###########################################csharp#############################################
def csharp_query_code(csharp_data,split_num):
    query_data=csharp_data['orgin_query'].tolist()
    query_split_list=[query_data[i:i+split_num] for i in range(0, len(query_data), split_num)]
    pool= ThreadPool(10)
    query_list=pool.map(multi_query_procsharp, query_split_list)
    pool.close()
    pool.join()
    query=[]
    for p in query_list:
        query+=p
    print('query处理条数：%d'%len(query))

    code_data=csharp_data['orgin_code'].tolist()
    code_split_list=[code_data[i:i+split_num] for i in range(0, len(code_data), split_num)]
    pool= ThreadPool(10)
    code_list=pool.map(multi_code_procsharp, code_split_list)
    pool.close()
    pool.join()
    code=[]
    for p in code_list:
        code+=p
    print('code处理条数：%d'%len(code))

    col_name =csharp_data.columns.tolist()

    #构建集
    csharp_data.insert(col_name.index('orgin_query')+1, 'parse_query', query)
    csharp_data.insert(col_name.index('orgin_code')+2, 'parse_code', code)

    return csharp_data

###########################################javang#############################################
def javang_query_code(javang_data,split_num):
    query_data=javang_data['orgin_query'].tolist()
    query_split_list=[query_data[i:i+split_num] for i in range(0,len(query_data), split_num)]
    pool= ThreadPool(10)
    query_list=pool.map(multi_query_projavang, query_split_list)
    pool.close()
    pool.join()
    query=[]
    for p in query_list:
        query+=p
    print('query处理条数：%d'%len(query))

    code_data=javang_data['orgin_code'].tolist()
    code_split_list=[code_data[i:i+split_num] for i in range(0,len(code_data), split_num)]
    pool= ThreadPool(10)
    code_list=pool.map(multi_code_projavang, code_split_list)
    pool.close()
    pool.join()
    code=[]
    for p in code_list:
        code+=p
    print('code处理条数：%d'%len(code))

    col_name = javang_data.columns.tolist()

    # 构建集
    javang_data.insert(col_name.index('orgin_query') + 1, 'parse_query', query)
    javang_data.insert(col_name.index('orgin_code') + 2, 'parse_code', code)

    return javang_data


###########################################javang#############################################
def python_query_code(python_data,split_num):
    query_data=python_data['orgin_query'].tolist()
    query_split_list=[query_data[i:i+split_num] for i in range(0,len(query_data), split_num)]
    pool= ThreadPool(10)
    query_list=pool.map(multi_query_propython, query_split_list)
    pool.close()
    pool.join()
    query=[]
    for p in query_list:
        query+=p
    print('query处理条数：%d'%len(query))

    code_data=python_data['orgin_code'].tolist()
    code_split_list=[code_data[i:i+split_num] for i in range(0,len(code_data), split_num)]
    pool= ThreadPool(10)
    code_list=pool.map(multi_code_propython, code_split_list)
    pool.close()
    pool.join()
    code=[]
    for p in code_list:
        code+=p
    print('code处理条数：%d'%len(code))

    col_name = python_data.columns.tolist()

    # 构建集
    python_data.insert(col_name.index('orgin_query') + 1, 'parse_query', query)
    python_data.insert(col_name.index('orgin_code') + 2, 'parse_code', code)

    return python_data


def read_fusecsv_data(file_path,csvda_type,lang_type,split_num,save_path):
    # 读取CSV数据
    lang_data=pd.read_csv(file_path+'%s_fuse_%s_np.csv'%(csvda_type,lang_type),index_col=0,sep='|',encoding='utf-8')
    # 数据统计
    print('%s类型的%s处理之前共有%d条数据'%(csvda_type,lang_type,len(lang_data)))   #76288条

    data_part1 = lang_data[:int(len(lang_data) / 3)]
    data_part2 = lang_data[int(len(lang_data) / 3) + 1:int(len(lang_data) * 2 / 3)]
    data_part3 = lang_data[int(len(lang_data) * 2 / 3) + 1:]

    if lang_type == 'csharp':

        parse_data1 = csharp_query_code(data_part1, split_num)
        parse_data2 = csharp_query_code(data_part2, split_num)
        parse_data3 = csharp_query_code(data_part3, split_num)

        # 合并数据
        parse_data = pd.concat([parse_data1, parse_data2, parse_data3], axis=0)

        if csvda_type == 'single':
            # 去除解析无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            lang_parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之前共有%d条数据' % (csvda_type,lang_type, len(lang_data)))  # 76199条
            # 保存数据
            lang_parse_data.to_csv(save_path + '%s_fuse_%s_parse.csv'%(csvda_type,lang_type), sep='|', encoding='utf-8')  # 保存行名
            print('%s类型的%s数据处理完毕！' %(csvda_type,lang_type))

        if csvda_type == 'mutiple':
            # 筛选排名0
            rank_data = parse_data[parse_data['code_rank'] == 0]
            # 取出ID
            null_idx = rank_data[rank_data['parse_code'] == '-1000']
            # 去除首位无效
            acce_idx = [id for id in parse_data['ID'].tolist() if id not in null_idx]
            # 抽取有效值
            parse_data = parse_data[parse_data['ID'].isin(acce_idx)]
            # 去除他位无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之后共有%d条数据' % (csvda_type,lang_type,len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path + '%s_fuse_%s_parse.csv'%(csvda_type,lang_type), sep='|', encoding='utf-8')  # 保存行名
            print('%s类型的%s数据处理完毕！' %(csvda_type, lang_type))

    if lang_type == 'sqlang':

        parse_data1 = sqlang_query_code(data_part1, split_num)
        parse_data2 = sqlang_query_code(data_part2, split_num)
        parse_data3 = sqlang_query_code(data_part3, split_num)

        # 合并数据
        parse_data = pd.concat([parse_data1, parse_data2, parse_data3], axis=0)

        if csvda_type == 'single':

            # 去除解析无效
            parse_data = parse_data[parse_data['parse_code'] !='-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之后共有%d条数据' % (csvda_type,lang_type,len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path+'%s_fuse_%s_parse.csv'%(csvda_type,lang_type), sep='|', encoding='utf-8')  #保存行名
            print('%s类型的%s数据处理完毕！'%(csvda_type,lang_type))

        if csvda_type == 'mutiple':

            # 筛选排名0
            rank_data = parse_data[parse_data['code_rank'] == 0]
            # 取出ID
            null_idx = rank_data[rank_data['parse_code'] =='-1000']
            # 去除首位无效
            acce_idx = [id for id in parse_data['ID'].tolist() if id not in null_idx]
            # 抽取有效值
            parse_data = parse_data[parse_data['ID'].isin(acce_idx)]
            # 去除他位无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之后共有%d条数据' % (csvda_type,lang_type,len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path + '%s_fuse_%s_parse.csv'%(csvda_type,lang_type), sep='|', encoding='utf-8') #保存行名
            print('%s类型的%s数据处理完毕！'%(csvda_type, lang_type))

    if lang_type == 'javang':

        parse_data1 = javang_query_code(data_part1, split_num)
        parse_data2 = javang_query_code(data_part2, split_num)
        parse_data3 = javang_query_code(data_part3, split_num)

        # 合并数据
        parse_data = pd.concat([parse_data1, parse_data2, parse_data3], axis=0)

        if csvda_type == 'single':
            # 去除解析无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之后共有%d条数据' % (csvda_type,lang_type,len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path + '%s_fuse_%s_parse.csv'%(csvda_type,lang_type), sep='|', encoding='utf-8') #保存行名
            print('%s类型的%s数据处理完毕！' % (csvda_type, lang_type))

        if csvda_type == 'mutiple':
            # 筛选排名0
            rank_one = parse_data[parse_data['code_rank'] == 0]
            # 取出ID
            null_idx = rank_one[rank_one['parse_code'] == '-1000']
            # 去除首位无效
            acce_idx = [id for id in parse_data['ID'].tolist() if id not in null_idx]
            # 抽取有效值
            parse_data = parse_data[parse_data['ID'].isin(acce_idx)]
            # 去除他位无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之后共有%d条数据' % (csvda_type,lang_type,len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path + '%s_fuse_%s_parse.csv'%(csvda_type,lang_type),sep='|', encoding='utf-8') #保存行名
            print('%s类型的%s数据处理完毕！' % (csvda_type, lang_type))

    if lang_type == 'python':

        parse_data1 = python_query_code(data_part1, split_num)
        parse_data2 = python_query_code(data_part2, split_num)
        parse_data3 = python_query_code(data_part3, split_num)

        # 合并数据
        parse_data = pd.concat([parse_data1, parse_data2, parse_data3], axis=0)

        if csvda_type == 'single':
            # 去除解析无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之前共有%d条数据' % (csvda_type,lang_type,len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path + '%s_fuse_%s_parse.csv' %(csvda_type,lang_type), sep='|', encoding='utf-8')  # 保存行名
            print('%s类型的%s数据处理完毕！' % (csvda_type, lang_type))

        if csvda_type == 'mutiple':
            # 筛选排名0
            rank_data = parse_data[parse_data['code_rank'] == 0]
            # 取出ID
            null_idx = rank_data[rank_data['parse_code'] == '-1000']
            # 去除首位无效
            acce_idx = [id for id in parse_data['ID'].tolist() if id not in null_idx]
            # 抽取有效值
            parse_data = parse_data[parse_data['ID'].isin(acce_idx)]
            # 去除他位无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之前共有%d条数据' % (csvda_type,lang_type, len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path + '%s_fuse_%s_parse.csv'%(csvda_type,lang_type), sep='|', encoding='utf-8')  # 保存行名
            print('%s类型的%s数据处理完毕！' % (csvda_type, lang_type))


def read_soqccsv_data(file_path,csvda_type,lang_type,split_num,save_path):
    # 读取CSV数据
    lang_data = pd.read_csv(file_path+'%s_soqc_%s_np.csv'%(csvda_type, lang_type),index_col=0, sep='|',encoding='utf-8')
    # 数据统计
    print('%s类型的%s处理之前共有%d条数据'%(csvda_type,lang_type,len(lang_data)))  #76288条

    data_part1 = lang_data[:int(len(lang_data) / 3)]
    data_part2 = lang_data[int(len(lang_data) / 3) + 1:int(len(lang_data) * 2 / 3)]
    data_part3 = lang_data[int(len(lang_data) * 2 / 3) + 1:]

    if lang_type == 'sqlang':

        parse_data1 = sqlang_query_code(data_part1, split_num)
        parse_data2 = sqlang_query_code(data_part2, split_num)
        parse_data3 = sqlang_query_code(data_part3, split_num)

        # 合并数据
        parse_data = pd.concat([parse_data1, parse_data2, parse_data3], axis=0)

        if csvda_type == 'single':

            # 去除解析无效
            parse_data = parse_data[parse_data['parse_code'] !='-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之后共有%d条数据' % (csvda_type,lang_type,len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path+'%s_soqc_%s_parse.csv'%(csvda_type,lang_type), sep='|', encoding='utf-8')  #保存行名
            print('%s类型的%s数据处理完毕！'%(csvda_type,lang_type))

        if csvda_type == 'mutiple':

            # 筛选排名0
            rank_data = parse_data[parse_data['code_rank'] == 0]
            # 取出ID
            null_idx= rank_data[rank_data['parse_code'] =='-1000']
            # 去除首位无效
            acce_idx = [id for id in parse_data['ID'].tolist() if id not in null_idx]
            # 抽取有效值
            parse_data = parse_data[parse_data['ID'].isin(acce_idx)]
            # 去除他位无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之后共有%d条数据' % (csvda_type,lang_type,len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path + '%s_soqc_%s_parse.csv'%(csvda_type,lang_type), sep='|', encoding='utf-8') #保存行名
            print('%s类型的%s数据处理完毕！'%(csvda_type, lang_type))

    if lang_type == 'python':

        parse_data1 = python_query_code(data_part1, split_num)
        parse_data2 = python_query_code(data_part2, split_num)
        parse_data3 = python_query_code(data_part3, split_num)

        # 合并数据
        parse_data = pd.concat([parse_data1, parse_data2, parse_data3], axis=0)

        if csvda_type == 'single':
            # 去除解析无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之后共有%d条数据' % (csvda_type,lang_type,len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path + '%s_soqc_%s_parse.csv' %(csvda_type,lang_type), sep='|', encoding='utf-8')  # 保存行名
            print('%s类型的%s数据处理完毕！' % (csvda_type, lang_type))

        if csvda_type == 'mutiple':
            # 筛选排名0
            rank_data = parse_data[parse_data['code_rank'] == 0]
            # 取出ID
            null_idx = rank_data[rank_data['parse_code'] == '-1000']
            # 去除首位无效
            acce_idx = [id for id in parse_data['ID'].tolist() if id not in null_idx]
            # 抽取有效值
            parse_data = parse_data[parse_data['ID'].isin(acce_idx)]
            # 去除他位无效
            parse_data = parse_data[parse_data['parse_code'] != '-1000']
            # 重新0-索引
            parse_data = parse_data.reset_index(drop=True)
            # 统计个数
            print('%s类型的%s处理之前共有%d条数据' % (csvda_type,lang_type, len(lang_data)))  # 76199条
            # 保存数据
            parse_data.to_csv(save_path + '%s_soqc_%s_parse.csv'%(csvda_type,lang_type), sep='|', encoding='utf-8')  # 保存行名
            print('%s类型的%s数据处理完毕！' % (csvda_type,lang_type))


#参数配置
file_path='../rank_corpus/'
save_path='../parse_corpus/'
split_num=1000

#数据类型
single_type='single'
mutiple_type='mutiple'

#语言类型
sqlang_type='sqlang'
csharp_type='csharp'
javang_type='javang'
python_type='python'


if __name__ == '__main__':
    ###################################fuse##############################

    # sqlang
    read_fusecsv_data(file_path,single_type,sqlang_type,split_num,save_path)
    read_fusecsv_data(file_path,mutiple_type,sqlang_type,split_num,save_path)
    # csharp
    read_fusecsv_data(file_path,single_type,csharp_type,split_num,save_path)
    read_fusecsv_data(file_path,mutiple_type,csharp_type,split_num,save_path)
    # javang
    read_fusecsv_data(file_path,single_type,javang_type,split_num,save_path)
    read_fusecsv_data(file_path,mutiple_type,javang_type,split_num,save_path)
    # python
    read_fusecsv_data(file_path,single_type,python_type,split_num,save_path)
    read_fusecsv_data(file_path,mutiple_type,python_type,split_num,save_path)

    ###################################fuse##############################
    # sqlang
    read_soqccsv_data(file_path,mutiple_type,sqlang_type,split_num,save_path)
    # python
    read_soqccsv_data(file_path,mutiple_type,python_type,split_num,save_path)


