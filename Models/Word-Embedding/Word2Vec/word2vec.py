from gensim.models import word2vec
"""
using gensim word2vec get vector
"""

"""
=================Train================
"""
# # 加载语料
# sentences = word2vec.Text8Corpus('./data/en_text8')
# # 训练模型
# model = word2vec.Word2Vec(sentences, size=200, window=5, min_count=3, workers=4)
# # 保存模型
# model.save('./model/en_text8.model')

"""
=================Load Model================
"""
# 加载模型
model = word2vec.Word2Vec.load('./model/en_text8.model')

print(model["a"], len(model["set"]))

# # 选出最相似的10个词
# for e in model.most_similar(positive=['set'], topn=10):
#    print(e[0], e[1])
