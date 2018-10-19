# https://www.tensorflow.org/tutorials/representation/word2vec#vector-representations-of-words
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# http://solarisailab.com/archives/374

# -*- coding: utf-8 -*-

# 절대 임포트 설정
from __future__ import absolute_import
from __future__ import print_function

# 필요한 라이브러리들을 임포트

import math


import numpy as np

import tensorflow as tf

from sources.tensorflow.model_base import ModelBase
from sources.tensorflow.hparams import HParams


class Word2Vec(ModelBase):
    def __init__(self, hparams):

        vocabulary_size = hparams.vocabulary_size

        # Step 4: skip-gram model 만들고 학습시킨다.

        batch_size = hparams.batch_size
        embedding_size = hparams.embedding_size  # embedding vector의 크기.

        # sample에 대한 validation set은 원래 랜덤하게 선택해야한다. 하지만 여기서는 validation samples을
        # 가장 자주 생성되고 낮은 숫자의 ID를 가진 단어로 제한한다.
        valid_size = hparams.valid_size  # validation 사이즈.
        valid_window = hparams.valid_window  # 분포의 앞부분(head of the distribution)에서만 validation sample을 선택한다.
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        num_sampled = hparams.num_sampled  # sample에 대한 negative examples의 개수.


        # 트레이닝을 위한 인풋 데이터들
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # embedding vectors 행렬을 랜덤값으로 초기화
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            # 행렬에 트레이닝 데이터를 지정
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # NCE loss를 위한 변수들을 선언
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # batch의 average NCE loss를 계산한다.
        # tf.nce_loss 함수는 loss를 평가(evaluate)할 때마다 negative labels을 가진 새로운 샘플을 자동적으로 생성한다.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size))

        # SGD optimizer를 생성한다.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # minibatch examples과 모든 embeddings에 대해 cosine similarity를 계산한다.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.loss = loss
        self.optimizer = optimizer
        self.similarity = similarity
        self.normalized_embeddings = normalized_embeddings
        self.valid_examples = valid_examples


    @staticmethod
    def get_default_params():
        """
        All parameters required to run the model. Can be changed by update function.

        :return:
        """
        return HParams(
            epochs          = 10,
            learning_rate   = 0.001,
            use_gpu         = True,
            batch_size      = 128,
            embedding_size  = 128,  # embedding vector의 크기.
            skip_window     = 1,  # 윈도우 크기 : 왼쪽과 오른쪽으로 얼마나 많은 단어를 고려할지를 결정.
            num_skips       = 2,  # 레이블(label)을 생성하기 위해 인풋을 얼마나 많이 재사용 할 것인지를 결정.
            vocabulary_size = 50000,
            # sample에 대한 validation set은 원래 랜덤하게 선택해야한다. 하지만 여기서는 validation samples을
            # 가장 자주 생성되고 낮은 숫자의 ID를 가진 단어로 제한한다.
            valid_size = 16,  # validation 사이즈.
            valid_window = 100,  # 분포의 앞부분(head of the distribution)에서만 validation sample을 선택한다.
            valid_examples = np.random.choice(100, 16, replace=False),
            num_sampled = 64,  # sample에 대한 negative examples의 개수.
            gpu=0
        )

    def get_train_list(self):
        return [self.optimizer, self.loss]

    def get_train_feed_dict(self, data, labels):
        return {self.train_inputs: data, self.train_labels: labels}



# Step 6: embeddings을 시각화한다.

# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#   assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#   plt.figure(figsize=(18, 18))  #in inches
#   for i, label in enumerate(labels):
#     x, y = low_dim_embs[i,:]
#     plt.scatter(x, y)
#     plt.annotate(label,
#                  xy=(x, y),
#                  xytext=(5, 2),
#                  textcoords='offset points',
#                  ha='right',
#                  va='bottom')
#
#   plt.savefig(filename)
#
# try:
#   from sklearn.manifold import TSNE
#   import matplotlib.pyplot as plt
#
#   tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#   plot_only = 500
#   low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
#   labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#   plot_with_labels(low_dim_embs, labels)
#
# except ImportError:
#   print("Please install sklearn and matplotlib to visualize embeddings.")