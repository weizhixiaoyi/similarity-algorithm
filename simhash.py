import numpy as np
import jieba.analyse


def get_keyword_participle(str):
    """get the correlative and tf-idf number.

    Parameters
    ----------
    str: str

    Returns
    -------
    correlative and weight
    """
    keyword_seq_list = jieba.analyse.extract_tags(str, topK=20, withWeight=True, allowPOS=())
    return keyword_seq_list


def get_string_hash(str):
    """change one  word list to hash str.

    Parameters
    ----------
    str: word

    Returns
    -------
    hash str
    """
    if str == "":
        return 0
    else:
        hash = ord(str[0]) << 7
        m = 1000003
        mask = 2 ** 128 - 1
        for s in str:
            hash = ((hash * m) ^ ord(s)) & mask
        hash ^= len(str)
        if hash == -1:
            hash = -2
        hash = bin(hash).replace('ob', '').zfill(64)[-64:]
        return hash


def get_simhash(keywordlist):
    """change word to hash str and caculate weight.

    Parameters
    ----------
    keywordlist: list

    Returns
    -------
    hash str
    """
    keyList = []
    for i in range(0, len(keywordlist)):
        feature = get_string_hash(keywordlist[i][0])
        weight = int(keywordlist[i][1] * 20)
        temp = []
        for i in feature:
            if i == '1':
                temp.append(weight)
            else:
                temp.append(-weight)
            keyList.append(temp)
    # print(keyList)
    wordList = np.sum(np.array(keyList), axis=0)
    if keyList == []:
        return '00'
    simhash_str = ''
    for i in wordList:
        if (i > 0):
            simhash_str = simhash_str + '1'
        else:
            simhash_str = simhash_str + '0'
    return simhash_str


def get_hamming_distance(simhashstr1, simhashstr2):
    """hamming distance.

    Parameters
    ----------
    str1: hash str1
    str2: hash str2

    Returns
    -------
    hamming distance
    """
    d1 = '0b' + simhashstr1
    d2 = '0b' + simhashstr2
    n = int(d1, 2) ^ int(d2, 2)
    diff = 0
    while n:
        n &= (n - 1)
        diff += 1
    return diff


def simhash_distance(str1, str2):
    """simhash distance.
    https://yanyiwu.com/work/2014/01/30/simhash-shi-xian-xiang-jie.html

    Parameters
    ----------
    str1: string1
    str2: string2

    Returns
    -------
    distance
    """
    # str1
    keywordlist1 = get_keyword_participle(str1)
    simhashstr1 = get_simhash(keywordlist1)
    # str2
    keywordlist2 = get_keyword_participle(str2)
    simhashstr2 = get_simhash(keywordlist2)
    # caculate simhash distance
    dis = get_hamming_distance(simhashstr1, simhashstr2)
    return dis


if __name__ == '__main__':
    str1 = '你妈妈喊你回家吃饭哦，回家罗回家罗'
    str2 = '你妈妈叫你回家吃饭啦，回家罗回家罗'
    dis = simhash_distance(str1, str2)
    print(dis)