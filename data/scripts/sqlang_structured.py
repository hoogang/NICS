# -*- coding: utf-8 -*-
# python2

import re
import sqlparse #0.2.4

#骆驼命名法
import inflection

#词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

#词干提取
from nltk.corpus import wordnet

#############################################################################
OTHER = 0
FUNCTION = 1
BLANK = 2
KEYWORD = 3
INTERNAL = 4

TABLE = 5
COLUMN = 6
INTEGER = 7
FLOAT = 8
HEX = 9
STRING = 10
WILDCARD = 11

SUBQUERY = 12

DUD = 13

ttypes = {0: "OTHER", 1: "FUNCTION", 2: "BLANK", 3: "KEYWORD", 4: "INTERNAL", 5: "TABLE", 6: "COLUMN", 7: "INTEGER",
          8: "FLOAT", 9: "HEX", 10: "STRING", 11: "WILDCARD", 12: "SUBQUERY", 13: "DUD", }

scanner = re.Scanner([(r"\[[^\]]*\]", lambda scanner, token: token), (r"\+", lambda scanner, token: "REGPLU"),
                      (r"\*", lambda scanner, token: "REGAST"), (r"%", lambda scanner, token: "REGCOL"),
                      (r"\^", lambda scanner, token: "REGSTA"), (r"\$", lambda scanner, token: "REGEND"),
                      (r"\?", lambda scanner, token: "REGQUE"),
                      (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),
                      (r'.', lambda scanner, token: None), ])

#---------------------子函数1：代码的规则--------------------
def tokenizeRegex(s):
    results = scanner.scan(s)[0]
    return results

#---------------------子函数2：代码的规则--------------------
class Sqlang_Lexer():
    @staticmethod
    def sanitizeSql(sql):
        s = sql.strip().lower()
        if not s[-1] == ";":
            s += ';'
        s = re.sub(r'\(', r' ( ', s)
        s = re.sub(r'\)', r' ) ', s)
        words = ['index', 'table', 'day', 'year', 'user', 'text']
        for word in words:
            s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)
            s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)
        s = s.replace('#', '')
        return s

    def parseStrings(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.parseStrings(c)
        elif tok.ttype == STRING:
            if self.regex:
                tok.value = ' '.join(tokenizeRegex(tok.value))
            else:
                tok.value = "CODSTR"

    def renameIdentifiers(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameIdentifiers(c)
        elif tok.ttype == COLUMN:
            if str(tok) not in self.idMap["COLUMN"]:
                colname = "col" + str(self.idCount["COLUMN"])
                self.idMap["COLUMN"][str(tok)] = colname
                self.idMapInv[colname] = str(tok)
                self.idCount["COLUMN"] += 1
            tok.value = self.idMap["COLUMN"][str(tok)]
        elif tok.ttype == TABLE:
            if str(tok) not in self.idMap["TABLE"]:
                tabname = "tab" + str(self.idCount["TABLE"])
                self.idMap["TABLE"][str(tok)] = tabname
                self.idMapInv[tabname] = str(tok)
                self.idCount["TABLE"] += 1
            tok.value = self.idMap["TABLE"][str(tok)]

        elif tok.ttype == FLOAT:
            tok.value = "CODFLO"
        elif tok.ttype == INTEGER:
            tok.value = "CODINT"
        elif tok.ttype == HEX:
            tok.value = "CODHEX"

    def __hash__(self):
        return hash(tuple([str(x) for x in self.tokensWithBlanks]))

    def __init__(self, sql, regex=False, rename=True):
        self.sql = Sqlang_Lexer.sanitizeSql(sql)

        self.idMap = {"COLUMN": {}, "TABLE": {}}
        self.idMapInv = {}
        self.idCount = {"COLUMN": 0, "TABLE": 0}
        self.regex = regex

        self.parseTreeSentinel = False
        self.tableStack = []

        self.parse = sqlparse.parse(self.sql)
        self.parse = [self.parse[0]]

        self.removeWhitespaces(self.parse[0])
        self.identifyLiterals(self.parse[0])
        self.parse[0].ptype = SUBQUERY
        self.identifySubQueries(self.parse[0])
        self.identifyFunctions(self.parse[0])
        self.identifyTables(self.parse[0])

        self.parseStrings(self.parse[0])

        if rename:
            self.renameIdentifiers(self.parse[0])

        self.tokens = Sqlang_Lexer.getTokens(self.parse)

    @staticmethod
    def getTokens(parse):
        flatParse = []
        for expr in parse:
            for token in expr.flatten():
                if token.ttype == STRING:
                    flatParse.extend(str(token).split(' '))
                else:
                    flatParse.append(str(token))
        return flatParse

    def removeWhitespaces(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            tmpChildren = []
            for c in tok.tokens:
                if not c.is_whitespace:
                    tmpChildren.append(c)

            tok.tokens = tmpChildren
            for c in tok.tokens:
                self.removeWhitespaces(c)

    def identifySubQueries(self, tokenList):
        isSubQuery = False

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                subQuery = self.identifySubQueries(tok)
                if (subQuery and isinstance(tok, sqlparse.sql.Parenthesis)):
                    tok.ttype = SUBQUERY
            elif str(tok) == "select":
                isSubQuery = True
        return isSubQuery

    def identifyLiterals(self, tokenList):
        blankTokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]
        blankTokenTypes = [sqlparse.sql.Identifier]

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                tok.ptype = INTERNAL
                self.identifyLiterals(tok)
            elif (tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select"):
                tok.ttype = KEYWORD
            elif (tok.ttype == sqlparse.tokens.Number.Integer or tok.ttype == sqlparse.tokens.Literal.Number.Integer):
                tok.ttype = INTEGER
            elif (tok.ttype == sqlparse.tokens.Number.Hexadecimal or tok.ttype == sqlparse.tokens.Literal.Number.Hexadecimal):
                tok.ttype = HEX
            elif (tok.ttype == sqlparse.tokens.Number.Float or tok.ttype == sqlparse.tokens.Literal.Number.Float):
                tok.ttype = FLOAT
            elif (tok.ttype == sqlparse.tokens.String.Symbol or tok.ttype == sqlparse.tokens.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Single or tok.ttype == sqlparse.tokens.Literal.String.Symbol):
                tok.ttype = STRING
            elif (tok.ttype == sqlparse.tokens.Wildcard):
                tok.ttype = WILDCARD
            elif (tok.ttype in blankTokens or isinstance(tok, blankTokenTypes[0])):
                tok.ttype = COLUMN

    def identifyFunctions(self, tokenList):
        for tok in tokenList.tokens:
            if (isinstance(tok, sqlparse.sql.Function)):
                self.parseTreeSentinel = True
            elif (isinstance(tok, sqlparse.sql.Parenthesis)):
                self.parseTreeSentinel = False
            if self.parseTreeSentinel:
                tok.ttype = FUNCTION
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyFunctions(tok)

    def identifyTables(self, tokenList):
        if tokenList.ptype == SUBQUERY:
            self.tableStack.append(False)

        for i in range(len(tokenList.tokens)):
            prevtok = tokenList.tokens[i - 1]
            tok = tokenList.tokens[i]

            if (str(tok) == "." and tok.ttype == sqlparse.tokens.Punctuation and prevtok.ttype == COLUMN):
                prevtok.ttype = TABLE

            elif (str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = True

            elif ((str(tok) == "where" or str(tok) == "on" or str(tok) == "group" or str(tok) == "order" or str(tok) == "union") and tok.ttype == sqlparse.tokens.Keyword):
                self.tableStack[-1] = False

            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyTables(tok)

            elif (tok.ttype == COLUMN):
                if self.tableStack[-1]:
                    tok.ttype = TABLE

        if tokenList.ptype == SUBQUERY:
            self.tableStack.pop()

    def __str__(self):
        return ' '.join([str(tok) for tok in self.tokens])

    def parsesql(self):
        return [str(tok) for tok in self.tokens]
#############################################################################

#############################################################################
#缩略词处理
def abbrev(line):
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
    # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")
    # s
    pat_s2 = re.compile("(?<=s)\"s?")
    # not
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")
    # would
    pat_would = re.compile("(?<=[a-zA-Z])\"d")
    # will
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")
    # am
    pat_am = re.compile("(?<=[I|i])\"m")
    # are
    pat_are = re.compile("(?<=[a-zA-Z])\"re")
    # have
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")

    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line

#获取词性
def get_pos(tag):
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

#---------------------子函数1：句子的去冗--------------------
def pro_query_line(line):
    # 句子预处理
    line = abbrev(line)
    line = line.strip()  # 去除首尾空格
    line = line.replace('\\n', ' ').replace('\n', ' ')
    # 骆驼命名转下划线
    line =inflection.underscore(line)
    #去除括号里内容
    space= re.compile(r"\([^\(|^\)]+\)")  # 后缀匹配
    line = re.sub(space, '', line)
    query_line=line.strip()
    return query_line

#---------------------子函数1：句子的分词--------------------
def pro_sent_word(sent_line):
    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    sent_line = re.sub(decimal, 'TAGINT', sent_line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    sent_line = re.sub(string, 'TAGSTR', sent_line)
    #替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    sent_line = re.sub(decimal, 'TAGINT', sent_line)
    #替换数字 56
    number = re.compile(r"\s\d+\s?")
    sent_line = re.sub(number, ' TAGINT ', sent_line)
    #替换字符 6c60b8e1
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")  # 后缀匹配
    sent_line = re.sub(other, 'TAGOER', sent_line)
    # 找单词
    cut_words = re.findall(r"[\w'-]+|[^\s\w]", sent_line)
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    #词性标注
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list=[]
    for word in cut_words:
        word_pos = get_pos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # 词性还原
            word = wnler.lemmatize(word, pos=word_pos)
        # 词干提取(效果最好）
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word.encode('utf8'))
    return word_list
#############################################################################

########################主函数：代码的tokens#################################
def sqlang_code_parse(line):
    try:
        line=line.strip()
        line = line.replace("\\n", "\n")
        query = Sqlang_Lexer(line, regex=True)
        typedCode = query.parsesql()
        typedCode = typedCode[:-1]
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')
        cut_tokens = [re.sub("\s+", " " , x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower().encode('utf8')  for x in cut_tokens]
        if token_list[0]:
            return  token_list
        else:
            return '-1000'
    except:
        return '-1000'
########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################
def sqlang_query_parse(line):
    query_line=pro_query_line(line)
    word_list=pro_sent_word(query_line)
    return word_list
#######################主函数：句子的tokens##################################

if __name__ == '__main__':
    print(sqlang_query_parse('How to create a (SQL Server function) to "join" multiple rows from a subquery into a single delimited field?'))
    print(sqlang_query_parse("how do i display records containing specific information in sql?"))
    print(sqlang_query_parse('MySQL Administrator Backups: "Compatibility Mode", What Exactly is this doing?'))

    print(sqlang_code_parse("db.users.find({"name": /.*m.*/})"))
    print(sqlang_code_parse('UPDATE Table1 \n SET Table1.col1 = Table2.col1 \n Table1.col2 = Table2.col2 FROM \n Table2 WHERE \n Table1.id = Table2.id'))
