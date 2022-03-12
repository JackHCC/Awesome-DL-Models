import time
import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


"""
=================== Load Data ====================

数据集使用的是来自Matt Mahoney的维基百科文章，数据集已经被清洗过，去除了特殊符号等，并不是全量数据，只是部分数据，所以实际上最后训练出的结果很一般（语料不够）。

如果想获取更全的语料数据，可以访问以下网站，这是gensim中Word2Vec提供的语料：
来自Matt Mahoney预处理后的文本子集(http://mattmahoney.net/dc/enwik9.zip)，里面包含了10亿个字符。
与第一条一样的经过预处理的文本数据(http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2)，但是包含了30个亿的字符。
多种语言的训练文本(http://www.statmt.org/wmt11/translation-task.html#download)。
UMBC webbase corpus(http://ebiquity.umbc.edu/redirect/to/resource/id/351/UMBC-webbase-corpus)
"""

with open('data/en_text8') as f:
    text = f.read()

"""
=================== 数据预处理 ====================

数据预处理过程主要包括：
1.替换文本中特殊符号并去除低频词
2.对文本分词
3.构建语料
4.单词映射表
"""


# 定义函数来完成数据的预处理
def preprocess(text_file, freq=5):
    """
    对文本进行预处理

    :param text_file: 文本数据
    :param freq: 词频阈值
    :return:
    """

    # 对文本中的符号进行替换
    text_file = text_file.lower()
    text_file = text_file.replace('.', ' <PERIOD> ')
    text_file = text_file.replace(',', ' <COMMA> ')
    text_file = text_file.replace('"', ' <QUOTATION_MARK> ')
    text_file = text_file.replace(';', ' <SEMICOLON> ')
    text_file = text_file.replace('!', ' <EXCLAMATION_MARK> ')
    text_file = text_file.replace('?', ' <QUESTION_MARK> ')
    text_file = text_file.replace('(', ' <LEFT_PAREN> ')
    text_file = text_file.replace(')', ' <RIGHT_PAREN> ')
    text_file = text_file.replace('--', ' <HYPHENS> ')
    text_file = text_file.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text_file = text_file.replace(':', ' <COLON> ')
    words_text = text_file.split()

    # 删除低频词，减少噪音影响
    word_counts = Counter(words_text)
    trimmed_words = [word for word in words_text if word_counts[word] > freq]

    return trimmed_words


# 清洗文本并分词
words = preprocess(text)
print(words[:20])

# 构建映射表
vocab = set(words)
vocab_to_int = {w: i for i, w in enumerate(vocab)}
int_to_vocab = {i: w for i, w in enumerate(vocab)}

print(type(vocab))

print("total words: {}".format(len(words)))
print("unique words: {}".format(len(set(words))))

# 对原文本进行vocab到int的转换
int_words = [vocab_to_int[w] for w in words]


"""
=================== 采样 ====================

对停用词进行采样，例如“the”， “of”以及“for”这类单词进行剔除。剔除这些单词以后能够加快我们的训练过程，同时减少训练过程中的噪音。

采用以下公式:
$$ P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} $$

其中$ t $是一个阈值参数，一般为1e-3至1e-5。  
$f(w_i)$ 是单词 $w_i$ 在整个数据集中的出现频次。  
$P(w_i)$ 是单词被删除的概率。

"""
# t值：阈值参数，一般为1e-3至1e-5
t = 1e-5
# 剔除概率阈值
threshold = 0.8

# 统计单词出现频次
int_word_counts = Counter(int_words)
total_count = len(int_words)
# 计算单词频率
word_freq = {w: c/total_count for w, c in int_word_counts.items()}
# 计算被删除的概率
prob_drop = {w: 1 - np.sqrt(t / word_freq[w]) for w in int_word_counts}

"""====== 原论文公式 ======"""
# 保留的概率公式
# prob_reserve = {w: (np.sqrt(word_freq[w] / 0.001) + 1) * (0.001 / word_freq[w]) for w in int_word_counts}
# prob_drop = 1 - prob_reserve

# 对单词进行采样
train_words = [w for w in int_words if prob_drop[w] < threshold]

print(len(train_words))


"""
=================== 构造batch ====================

Skip-Gram模型是通过输入词来预测上下文。因此我们要构造我们的训练样本。

对于一个给定词，离它越近的词可能与它越相关，离它越远的词越不相关，这里设置窗口大小为5，对于每个训练单词，
我们还会在[1:5]之间随机生成一个整数R，用R作为最终选择output word的窗口大小。这里之所以多加了一步随机数的窗口重新选择步骤，
是为了能够让模型更聚焦于当前input word的邻近词。
"""


def get_targets(words_text, idx, window_size=5):
    """
    获得input word的上下文单词列表

    :param words_text: 单词列表
    :param idx: input word的索引号
    :param window_size: 窗口大小
    :return: input word的上下文单词列表
    """
    target_window = np.random.randint(1, window_size + 1)
    # 这里要考虑input word前面单词不够的情况
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    # output words(即窗口中的上下文单词)
    targets = set(words_text[start_point: idx] + words_text[idx + 1: end_point + 1])
    return list(targets)


def get_batches(words_text, batch_size, window_size=5):
    """
    构造一个获取batch的生成器

    :param words_text:
    :param batch_size:
    :param window_size:
    :return:
    """
    n_batches = len(words_text) // batch_size

    # 仅取full batches
    words_text = words_text[:n_batches * batch_size]

    for idx in range(0, len(words_text), batch_size):
        x, y = [], []
        batch = words_text[idx: idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            # 由于一个input word会对应多个output word，因此需要长度统一
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y


"""
=================== 构建网络 ====================

该部分主要包括：
1.输入层
2.Embedding
3.Negative Sampling
"""
# Inputs Layer
train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

# Embedding Layer
# 嵌入矩阵的矩阵形状为  𝑣𝑜𝑐𝑎𝑏_𝑠𝑖𝑧𝑒×ℎ𝑖𝑑𝑑𝑒𝑛_𝑢𝑛𝑖𝑡𝑠_𝑠𝑖𝑧𝑒
# TensorFlow中的tf.nn.embedding_lookup函数可以实现lookup的计算方式
vocab_size = len(int_to_vocab)

# 嵌入维度 一般取：50-300
embedding_size = 200

with train_graph.as_default():
    # 嵌入层权重矩阵
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    # 实现lookup
    embed = tf.nn.embedding_lookup(embedding, inputs)

# Negative Sampling
# 负采样主要是为了解决梯度下降计算速度慢的问题。
# TensorFlow中的tf.nn.sampled_softmax_loss会在softmax层上进行采样计算损失，计算出的loss要比full softmax loss低。
n_sampled = 100

with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(vocab_size))

    # 计算negative sampling下的损失
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)


"""
=================== 验证 ====================

为了更加直观的看到训练的结果，将查看训练出的相近语义的词。
"""
with train_graph.as_default():
    # 随机挑选一些单词
    valid_size = 16
    valid_window = 100
    # 从不同位置各选8个单词
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size // 2))

    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))


epochs = 10  # 迭代轮数
batch_size = 1000  # batch大小
window_size = 10  # 窗口大小

with train_graph.as_default():
    saver = tf.train.Saver()  # 文件存储

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        #
        for x, y in batches:

            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            # 计算相似的词
            if iteration % 1000 == 0:
                # 计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8  # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1

    save_path = saver.save(sess, "checkpoints/en_text8.ckpt")
    embed_mat = sess.run(normalized_embedding)


viz_words = 500
tsne = TSNE()
embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])

fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
