# Author: Isaac Sim <gilgarad@igsinc.co.kr, wieryeveska@hotmail.com>
# Modified from http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


class LDA:
    def __init__(self, data):
        # self.input_path = input_path
        self.data = data

        self.n_samples = len(data) # number of samples
        self.n_features = 1000 # max features to use
        self.n_components = 10 # number of topics
        self.n_top_words = 20 # number of words per topic

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()

    def tfidf_vectorizer(self):
        # Use tf-idf features for NMF.
        print("Extracting tf-idf features for NMF...")
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=self.n_features,
                                           stop_words='english')
        t0 = time()
        self.tfidf = self.tfidf_vectorizer.fit_transform(self.data)
        print("done in %0.3fs." % (time() - t0))

    def tf_vectorizer(self):
        # Use tf (raw term count) features for LDA.
        print("Extracting tf features for LDA...")
        self.tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=self.n_features,
                                        stop_words='english')
        t0 = time()
        self.tf = self.tf_vectorizer.fit_transform(self.data)
        print("done in %0.3fs." % (time() - t0))
        print()

    def nmf_model(self):
        # Fit the NMF model
        print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
              "n_samples=%d and n_features=%d..."
              % (self.n_samples, self.n_features))
        t0 = time()
        nmf = NMF(n_components=self.n_components, random_state=1,
                  alpha=.1, l1_ratio=.5).fit(self.tfidf)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in NMF model (Frobenius norm):")
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names()
        self.print_top_words(nmf, tfidf_feature_names, self.n_top_words)

    def nmf_model2(self):
        # Fit the NMF model
        print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
              "tf-idf features, n_samples=%d and n_features=%d..."
              % (self.n_samples, self.n_features))
        t0 = time()
        nmf = NMF(n_components=self.n_components, random_state=1,
                  beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                  l1_ratio=.5).fit(self.tfidf)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names()
        self.print_top_words(nmf, tfidf_feature_names, self.n_top_words)

    def lda_model(self):
        print("Fitting LDA models with tf features, "
              "n_samples=%d and n_features=%d..."
              % (self.n_samples, self.n_features))
        lda = LatentDirichletAllocation(n_components=self.n_components, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        t0 = time()
        lda.fit(self.tf)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in LDA model:")
        tf_feature_names = self.tf_vectorizer.get_feature_names()
        self.print_top_words(lda, tf_feature_names, self.n_top_words)