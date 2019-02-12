import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.linalg import norm


def tf_cosine_similarity(str1, str2):
    """tf cosine similarity

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
    tf_cos = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    return tf_cos


def tfidf_consine_similarity(str1, str2):
    """tfidf cosine similarity

    Parameters
    ----------
    str1: string1
    str2: string2

    Returns
    -------
    similarity
    """
    s1, s2 = ' '.join(list(str1)), ' '.join(list(str2))
    corpus = [s1, s2]
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    vector = cv.fit_transform(corpus).toarray()
    transform = TfidfTransformer()
    tfidf = transform.fit_transform(vector).toarray()
    vector1, vector2 = tfidf[0][::-1], tfidf[1][::-1]
    tfidf_cos = np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    return tfidf_cos


def word_in_str_similarity(str1, str2):
    """tfidf cosine similarity

    Parameters
    ----------
    str1: string1
    str2: string2

    Returns
    -------
    similarity
    """
    str1 = ''.join(set(str1.replace(' ', '')))
    str2 = ''.join(set(str2.replace(' ', '')))
    if len(str1) > len(str2):
        str1, str2 = str2, str1
    common_word = 0
    for i in range(0, len(str1)):
        if str1[i] in str2:
            common_word += 1
    similarity = common_word / len(str2)
    return similarity


if __name__ == '__main__':
    str1 = '车子进水了 怎么办'
    str2 = '车子进水了 怎么办 啊啊啊啊啊啊'
    ans1 = tf_cosine_similarity(str1, str2)
    ans2 = tfidf_consine_similarity(str1, str2)
    ans3 = word_in_str_similarity(str1, str2)
    print(ans1)
    print(ans2)
    print(ans3)
