# import tensorflow as tf
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def load_model(path):
    session = tf.Session()
    checkpoint_dir = os.path.abspath(os.path.join(path, "checkpoints"))
    saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, "en_text8.ckpt.meta"))
    saver.restore(session, tf.train.latest_checkpoint('./checkpoints'))

    return session


def get_trainable_variables(session):
    tvs = [v for v in tf.trainable_variables()]
    print('获得所有可训练变量的权重:')
    for v in tvs:
        print(v.name)
        print(session.run(v))
        print(session.run(v).shape)


def get_global_variables():
    gv = [v for v in tf.global_variables()]
    print('获得所有变量:')
    for v in gv:
        print(v.name, '\n')


def get_operations():
    # sess.graph.get_operations()可以换为tf.get_default_graph().get_operations()
    ops = [o for o in session.graph.get_operations()]
    print('获得所有operations相关的tensor:')
    for o in ops:
        print(o.name, '\n')


def word2vec_from_text8(word, session):
    word_dict_text8 = np.load("dict/word_dict_text8.npy", allow_pickle=True)
    word_dict_text8 = word_dict_text8.tolist()
    word_index = word_dict_text8.index(word)

    embed_mat = session.run("Variable:0")

    return embed_mat[word_index, :], word_index


if __name__ == "__main__":
    ckpt_path = "./"
    session = load_model(ckpt_path)

    # get_global_variables()
    # get_trainable_variables(session)

    vec, index = word2vec_from_text8("set", session)
    print(vec, index)




