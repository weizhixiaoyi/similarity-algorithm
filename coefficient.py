import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance


def jaccard_coefficient(str1, str2):
    """jaccard coefficient.

    Parameters
    ----------
    str1: string1
    str2: string2

    Returns
    -------
    similarity
    """
    s1, s2 = ' '.join(list(str1)), ' '.join(list(str2))
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    similar = 1.0 * numerator / denominator
    return similar


def dice_coefficient(str1, str2):
    """dice coefficient.

    Parameters
    ----------
    str1: string1
    str2: string2

    Returns
    -------
    similarity
    """
    s1, s2 = ' '.join(list(str1)), ' '.join(list(str2))
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    vector1, vector2 = vectors[0], vectors[1]
    dis = distance.dice(vector1, vector2)
    return 1 - dis


if __name__ == '__main__':
    str1 = '车子进水了怎么办'
    str2 = '哎呀妈呀！车子进水了怎么办啊！'
    ans = dice_coefficient(str1, str2)
    print(ans)
