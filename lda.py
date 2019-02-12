from gensim import corpora, models, similarities
import logging
import jieba

def LdaModel(query, documents):
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    texts = [[word for word in jieba.cut(document, cut_all=False)] for document in documents]
    dictionary = corpora.Dictionary(texts)
    # word must appear >10 times, and no more than 40% documents
    dictionary.filter_extremes(no_below=40, no_above=0.1)
    print(dictionary)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100, iterations=500)
    query_bow = dictionary.doc2bow(jieba.cut(query, cut_all=False))
    

if __name__ == '__main__':
    documents = ['1、事故发生后，在未依法采取措施的情况下驾驶被保险机动车或者遗弃被保险机动车离开事故现场；',
    '2、饮酒、吸食或注射毒品、服用国家管制的精神药品或者麻醉药品；',
    '3、无驾驶证，驾驶证被依法扣留、暂扣、吊销、注销期间；',
    '4、驾驶与驾驶证载明的准驾车型不相符合的机动车；',
    '5、实习期内驾驶公共汽车、营运客车或者执行任务的警车、载有危险物品的机动车或牵引挂车的机动车；',
    '6、驾驶出租机动车或营业性机动车无交通运输管理部门核发的许可证书或其他必备证书；',
    '7、学习驾驶时无合法教练员随车指导； ',
    '8、非被保险人允许的驾驶人；']
    query = '没有驾驶证发生驾驶事故？'
    ans = LdaModel(query, documents)
    print(ans)
