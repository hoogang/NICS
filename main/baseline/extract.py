import re
#软件工程
import inflection

#词性还原
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

#词干提取
from nltk.stem import SnowballStemmer
stmer = SnowballStemmer("english")

#停用词处理
from nltk.corpus import stopwords


#获取词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


#缩略词处理
def abbrev(line):
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    #'s
    pat_s1 = re.compile("(?<=[a-zA-Z])\'s")
    #s
    pat_s2 = re.compile("(?<=s)\'s?")
    #not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    #would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    #will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    #am
    pat_am = re.compile("(?<=[I|i])\'m")
    #are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    #have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)
    new_line = line.replace('\'', ' ')

    return new_line


#程序类型分词
def parse(text):
    #缩略词处理
    abbrev_text = abbrev(text)
    # 获取词性
    cut_tokens = word_tokenize(abbrev_text)  # 分词
    tagged_sent = pos_tag(cut_tokens)  # 词性
    #处理骆驼命名法
    under_text = inflection.underscore(abbrev_text)
    #分拆单词a_b=a b
    split_text=under_text.replace('_',' ')
    #分词处理
    words_find = [w for w in re.findall(r"[a-z]_?\w+|\w{2,}", split_text)]
    #共有词汇
    common_words= list(set(cut_tokens).intersection(set(words_find)))
    #去除停用词
    filter_words = [ i for i in words_find if i not in stopwords.words('english')]
    #抽取标签
    common_tags = [tag for tag in tagged_sent if tag[0] in common_words]
    tags_dict = dict(common_tags)
    print(tags_dict)
    #词性还原
    wml_words = []
    for token in filter_words:
        if token in common_words:
            word_pos = get_wordnet_pos(tags_dict[token]) or wordnet.NOUN
            token = wnler.lemmatize(token, pos=word_pos)
        wml_words.append(token)
    stem_words = [stmer.stem(i) for i in wml_words]
    return  stem_words


#程序分词测试
if __name__ == '__main__':
    print(parse("leaves, Just the item padding sorter adds listView1.ListViewItemSorter = new ListViewItemComparer(columnToBeSortedBy)"))
