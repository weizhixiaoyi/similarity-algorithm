import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def edit_distance(str1, str2):
    """edit distance.

    Parameters
    ----------
    str1: string1
    str2: string2

    Returns
    -------
    distance
    """
    m, n = len(str1), len(str2)
    if m == 0: return n
    if n == 0: return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1): dp[i][0] = i
    for j in range(1, n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + 1, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    return float(dp[m][n])


def euclidean_distance(str1, str2):
    """euclidean distance.

    Parameters
    ----------
    str1: string1
    str2: string2

    Returns
    -------
    distance
    """
    s1, s2 = ' '.join(list(str1)), ' '.join(list(str2))
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    vector1 = np.mat(vectors[0])
    vector2 = np.mat(vectors[1])
    dis = np.sqrt((vector1 - vector2) * ((vector1 - vector2).T))
    return float(dis)


def manhattan_distance(str1, str2):
    """manhattan distance.

    Parameters
    ----------
    str1: string1
    str2: string2

    Returns
    -------
    distance
    """
    s1, s2 = ' '.join(list(str1)), ' '.join(list(str2))
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    vector1 = np.mat(vectors[0])
    vector2 = np.mat(vectors[1])
    dis = np.sum(np.abs(vector1 - vector2))
    return float(dis)


def jaro_distance(str1, str2):
    """jaro distance.

    Parameters
    ----------
    str1: string1
    str2: string2

    Returns
    -------
    distance
    """
    if len(str1) > len(str2):
        longStr = str1
        shortStr = str2
    else:
        longStr = str2
        shortStr = str1
    allowRange = (len(longStr) // 2) - 1
    mappingIndices = [-1] * len(shortStr)
    longMatch, shortMatch = [], []
    matches = 0
    for i in range(0, len(shortStr)):
        for j in range(max(0, i - allowRange), min(len(longStr), i + allowRange + 1)):
            if shortStr[i] == longStr[j]:
                matches = matches + 1
                mappingIndices[i] = j
                shortMatch.append(shortStr[i])
                longMatch.insert(j, shortStr[i])
                break
    halfTransPosition = 0
    for i in range(0, len(shortMatch)):
        if (mappingIndices[i] != i) and (shortMatch[i] != longMatch[i]):
            halfTransPosition += 1
    dis = 0
    if matches != 0:
        dis = ((matches / len(longStr)) + (matches / len(shortStr)) +
               ((matches - (halfTransPosition // 2)) / matches)) / 3

    return float(dis)


def jaro_winkler_distance(str1, str2):
    jaro = jaro_distance(str1, str2)
    prefix = 0
    for i in range(0, 4):
        if str1[i] == str2[i]:
            prefix += 1
        else:
            break
    dis = 0
    if (jaro > 0.7):
        dis = jaro + ((prefix * 0.1) * (1 - jaro))
    else:
        dis = jaro
    return float(dis)


if __name__ == '__main__':
    str1 = '你妈妈喊你回家吃饭哦，回家罗回家罗'
    str2 = '你妈妈叫你回家吃饭啦，回家罗回家罗'
    ans = jaro_winkler_distance(str1, str2)
    print(ans)
