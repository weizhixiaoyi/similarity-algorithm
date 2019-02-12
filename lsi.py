# encoding=utf-8
import jieba
from gensim import corpora, similarities, models


def LsiModel(query, docunment):
    corpora_documents = []
    for text in docunment:
        text_seg = list(jieba.cut(text))
        corpora_documents.append(text_seg)
    dictionary = corpora.Dictionary(corpora_documents)
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    # LSI model
    lsi = models.LsiModel(corpus_tfidf)
    corpus_lsi = lsi[corpus_tfidf]
    similarity_lsi = similarities.Similarity('Similarity-LSI-index', corpus_lsi, num_features=400, num_best=3)
    # query Solve
    query_list = list(jieba.cut(query))
    query_corpus = dictionary.doc2bow(query_list)
    query_tfidf = tfidf_model[query_corpus]
    query_lsi = lsi[query_tfidf]
    ans = similarity_lsi[query_lsi]
    return ans


if __name__ == '__main__':
    docunment = ['1、事故发生后，在未依法采取措施的情况下驾驶被保险机动车或者遗弃被保险机动车离开事故现场；',
                 '2、饮酒、吸食或注射毒品、服用国家管制的精神药品或者麻醉药品；',
                 '3、无驾驶证，驾驶证被依法扣留、暂扣、吊销、注销期间；',
                 '4、驾驶与驾驶证载明的准驾车型不相符合的机动车；',
                 '5、实习期内驾驶公共汽车、营运客车或者执行任务的警车、载有危险物品的机动车或牵引挂车的机动车；',
                 '6、驾驶出租机动车或营业性机动车无交通运输管理部门核发的许可证书或其他必备证书；',
                 '7、学习驾驶时无合法教练员随车指导； ',
                 '8、非被保险人允许的驾驶人；']
    query = '没有驾驶证发生驾驶事故？'
    ans = LsiModel(query, docunment)
    print(ans)
