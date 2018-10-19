import sys
from .data.data_loader import DataLoader
from datetime import datetime
from os.path import exists, join
from os import makedirs

from .model_saver import ModelSaver
from .models.word2vec import Word2Vec

# temporarily added for word2vec

from six.moves import xrange  # pylint: disable=redefined-builtin
import os
import zipfile
from six.moves import urllib


class Runner(ModelSaver):

    def get_model(self, model_name):

        models = {
            'word2vec': Word2Vec
        }

        if model_name not in models.keys():
            print('No matching model found')
            sys.exit()

        model = models[model_name]

        return model

    def train(self, data_path, test_path, output_path, model_name, hparams=None):

        # 1. Model Load
        self._model = self.get_model(model_name)

        default_hparams = self._model.get_default_params()
        if hparams is not None:
            # default_hparams.update_merge(hparams=hparams)
            hparams = default_hparams
        else:
            hparams = default_hparams

        model, sess, g = self._model_init(model=self._model, hparams=hparams)

        epochs = hparams.epochs
        batch_size = hparams.batch_size
        learning_rate = hparams.learning_rate

        # 2. Data Load
        data_loader = DataLoader(data_path=data_path, test_path=test_path, output_path=output_path, hparams=hparams)

        # label_list = data_loader.label_list
        # hparams.update(
        #     num_labels=len(label_list)
        # )
        # print('Label Length: %i' % (len(label_list)))
        ###

        # 3. Train
        global_step = 0
        print_step_interval = 500
        step_time = datetime.now()

        highest_accuracy = 0
        early_stop_count = 0

        for epoch in range(epochs):

            data_loader.reshuffle()
            avg_loss = 0.0
            avg_accuracy = 0.0

            for i, (data, labels) in enumerate(data_loader.batch_loader(data_loader.dataset, batch_size)):
                # print('Input X')
                # print(data)
                # print('Output Y')
                # print(labels)

                train_list = model.get_train_list()
                train_feed_dict = model.get_train_feed_dict(data, labels)
                _, loss = sess.run(train_list, feed_dict=train_feed_dict)

                avg_loss += float(loss)
                # avg_accuracy += float(accuracy)
                global_step += 1

                if global_step % print_step_interval == 0:
                    print('[global_step-%i] duration: %is train_loss: %f accuracy: %f' % (
                        global_step, (datetime.now() - step_time).seconds,
                        float(avg_loss / print_step_interval),
                        float(avg_accuracy / print_step_interval)))
                    avg_loss = 0
                    avg_accuracy = 0
                    step_time = datetime.now()

                if global_step % (print_step_interval * 10) == 0 and data_loader.test_dataset is not None:

                    step_t_time = datetime.now()
                    t_avg_loss = 0.0
                    t_avg_accuracy = 0.0
                    t_batch_iter_max = len(data_loader.test_dataset) / batch_size + 1

                    for t_i, (t_data, t_labels) in enumerate(
                            data_loader.batch_loader(data_loader.test_dataset, batch_size)):
                        accuracy, logits, loss = sess.run([model.accuracy, model.logits, model.loss],
                                                          feed_dict={model.x: t_data, model.y: t_labels,
                                                                     model.dropout_keep_prob: 1.0})

                        t_avg_loss += float(loss)
                        t_avg_accuracy += float(accuracy)

                    t_avg_loss = float(t_avg_loss / t_batch_iter_max)
                    t_avg_accuracy = float(t_avg_accuracy / t_batch_iter_max)
                    current_accuracy = t_avg_accuracy

                    print('[global_step-%i] duration: %is test_loss: %f accuracy: %f' % (global_step,
                                                                                         (
                                                                                                 datetime.now() - step_t_time).seconds,
                                                                                         t_avg_loss, t_avg_accuracy))

                    if highest_accuracy < current_accuracy:
                        print('Saving model...')
                        highest_accuracy = current_accuracy
                        current_accuracy = 0
                        if output_path is not None:
                            if not exists(output_path):
                                makedirs(output_path)
                        output_full_path = join(output_path,
                                                'loss%f_acc%f_epoch%i' % (avg_loss, avg_accuracy, epoch + 1))
                        self.save_session(directory=output_full_path, global_step=global_step)

                    if current_accuracy != 0:
                        early_stop_count += 1

                    step_time = datetime.now()

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                # CROSS VALIDATION PART?? EVALUATION
                if global_step % 10000 == 0:
                    sim = model.similarity.eval(session=sess)
                    reverse_dictionary = data_loader.dataset.reverse_dictionary
                    for i in xrange(hparams.valid_size):
                        valid_word = reverse_dictionary[model.valid_examples[i]]
                        top_k = 8  # nearest neighbors의 개수
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)

            if early_stop_count > 2:
                learning_rate = learning_rate * 0.90

            if early_stop_count > 5:
                print('Early stopped !')
                break

        final_embeddings = model.normalized_embeddings.eval()




    def train_original_b(self, data_path, test_path, output_path, model_name, hparams=None):

        # 1. Model Load
        self._model = self.get_model(model_name)

        default_hparams = self._model.get_default_params()
        if hparams is not None:
            default_hparams.update_merge(hparams=hparams)
            hparams = default_hparams
        else:
            hparams = default_hparams

        model, sess, g = self._model_init(model=self._model, hparams=hparams)

        epochs = hparams.epochs
        batch_size = hparams.batch_size
        learning_rate = hparams.learning_rate

        # 2. Data Load
        data_loader = DataLoader(data_path=data_path, test_path=test_path, output_path=output_path, hparams=hparams)

        label_list = data_loader.label_list
        hparams.update(
            num_labels=len(label_list)
        )
        print('Label Length: %i' % (len(label_list)))

        # 3. Train
        global_step = 0
        print_step_interval = 500
        step_time = datetime.now()

        highest_accuracy = 0
        early_stop_count = 0

        for epoch in range(epochs):

            data_loader.reshuffle()
            avg_loss = 0.0
            avg_accuracy = 0.0

            for i, (data, labels) in enumerate(data_loader.batch_loader(data_loader.dataset, batch_size)):
                # print(labels)
                # print(data, labels)
                _, loss, accuracy, logits, outputs = sess.run([model.train, model.loss, model.accuracy, model.logits,
                                                               model.outputs],
                                                              feed_dict={model.x: data, model.y: labels,
                                                                         model.dropout_keep_prob: 0.5,
                                                                         model.learning_rate: learning_rate
                                                                         })

                avg_loss += float(loss)
                avg_accuracy += float(accuracy)
                global_step += 1

                if global_step % print_step_interval == 0:
                    print('[global_step-%i] duration: %is train_loss: %f accuracy: %f' % (
                        global_step, (datetime.now() - step_time).seconds,
                        float(avg_loss / print_step_interval),
                        float(avg_accuracy / print_step_interval)))
                    avg_loss = 0
                    avg_accuracy = 0
                    step_time = datetime.now()

                if global_step % (print_step_interval * 10) == 0:

                    step_t_time = datetime.now()
                    t_avg_loss = 0.0
                    t_avg_accuracy = 0.0
                    t_batch_iter_max = len(data_loader.test_dataset) / batch_size + 1

                    for t_i, (t_data, t_labels) in enumerate(
                            data_loader.batch_loader(data_loader.test_dataset, batch_size)):
                        accuracy, logits, loss = sess.run([model.accuracy, model.logits, model.loss],
                                                          feed_dict={model.x: t_data, model.y: t_labels,
                                                                     model.dropout_keep_prob: 1.0})

                        t_avg_loss += float(loss)
                        t_avg_accuracy += float(accuracy)

                    t_avg_loss = float(t_avg_loss / t_batch_iter_max)
                    t_avg_accuracy = float(t_avg_accuracy / t_batch_iter_max)
                    current_accuracy = t_avg_accuracy

                    print('[global_step-%i] duration: %is test_loss: %f accuracy: %f' % (global_step,
                                                                                         (
                                                                                                     datetime.now() - step_t_time).seconds,
                                                                                         t_avg_loss, t_avg_accuracy))

                    if highest_accuracy < current_accuracy:
                        print('Saving model...')
                        highest_accuracy = current_accuracy
                        current_accuracy = 0
                        if output_path is not None:
                            if not exists(output_path):
                                makedirs(output_path)
                        output_full_path = join(output_path,
                                                'loss%f_acc%f_epoch%i' % (avg_loss, avg_accuracy, epoch + 1))
                        self.save_session(directory=output_full_path, global_step=global_step)

                    if current_accuracy != 0:
                        early_stop_count += 1

                    step_time = datetime.now()

            if early_stop_count > 2:
                learning_rate = learning_rate * 0.90

            if early_stop_count > 5:
                print('Early stopped !')
                break

    def train_new_b(self, model_name, hparams=None):

        # 1. Model Load
        self._model = self.get_model(model_name)

        default_hparams = self._model.get_default_params()
        if hparams is not None:
            default_hparams.update_merge(hparams=hparams)
            hparams = default_hparams
        else:
            hparams = default_hparams

        model, sess, g = self._model_init(model=self._model, hparams=hparams)

        epochs = hparams.epochs
        batch_size = hparams.batch_size
        learning_rate = hparams.learning_rate


        # 2. Data Load
        num_steps = 100001
        vocabulary_size = 50000

        input_path = None
        if input_path is None:
            words = self.samples()
        else:
            # not implemented yet
            words = self.read_data(filename=None)

        # data in words
        data, count, dictionary, reverse_dictionary = self.build_dataset(words, vocabulary_size)
        # del words  # Hint to reduce memory.

        print('Most common words (+UNK)', count[:5])
        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

        self.data_index = 0

        batch, labels = self.generate_batch(batch_size=8, num_skips=2, skip_window=1, data=data)
        for i in range(8):
            print(batch[i], reverse_dictionary[batch[i]],
                  '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
        print('Input X')
        print(batch)
        print('Output Y')
        print(labels)

        # 3. Train
        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = self.generate_batch(
                hparams.batch_size, hparams.num_skips, hparams.skip_window, data)
            feed_dict = {model.train_inputs: batch_inputs, model.train_labels: batch_labels}

            # optimizer op을 평가(evaluating)하면서 한 스텝 업데이트를 진행한다.
            _, loss_val = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # 평균 손실(average loss)은 지난 2000 배치의 손실(loss)로부터 측정된다.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = model.similarity.eval(session=sess)
                for i in xrange(hparams.valid_size):
                    valid_word = reverse_dictionary[model.valid_examples[i]]
                    top_k = 8  # nearest neighbors의 개수
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = model.normalized_embeddings.eval()



    # def samples(self):
    #     # Step 1: 필요한 데이터를 다운로드한다.
    #     url = 'http://mattmahoney.net/dc/'
    #     filename = 'text8.zip'
    #     expected_bytes = 31344016
    #
    #
    #     """파일이 존재하지 않으면 다운로드하고 사이즈가 적절한지 체크한다."""
    #     if not os.path.exists(filename):
    #         filename, _ = urllib.request.urlretrieve(url + filename, filename)
    #     statinfo = os.stat(filename)
    #     print(statinfo.st_size, expected_bytes)
    #     if statinfo.st_size == expected_bytes:
    #         print('Found and verified', filename)
    #     else:
    #         raise Exception(
    #             'Failed to verify ' + filename + '. Can you get to it with a browser?')
    #
    #     """zip파일 압축을 해제하고 단어들의 리스트를 읽는다."""
    #     with zipfile.ZipFile(filename) as f:
    #         words = f.read(f.namelist()[0]).split()
    #
    #     words = [w.decode("utf-8") for w in words]
    #
    #     print('Data size', len(words))
    #
    #     return words

    # def build_dataset(self, words, vocabulary_size):
    #     # Step 2: dictionary를 만들고 UNK 토큰을 이용해서 rare words를 교체(replace)한다.
    #     count = [['UNK', -1]]
    #     count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    #     dictionary = dict()
    #     for word, _ in count:
    #         dictionary[word] = len(dictionary)
    #     data = list()
    #     unk_count = 0
    #     for word in words:
    #         if word in dictionary:
    #             index = dictionary[word]
    #         else:
    #             index = 0  # dictionary['UNK']
    #             unk_count += 1
    #         data.append(index)
    #     count[0][1] = unk_count
    #     reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #     return data, count, dictionary, reverse_dictionary


    # def generate_batch(self, batch_size, num_skips, skip_window, data):
    #     # Step 3: skip-gram model을 위한 트레이닝 데이터(batch)를 생성하기 위한 함수.
    #     # global data_index
    #     assert batch_size % num_skips == 0
    #     assert num_skips <= 2 * skip_window
    #     batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    #     labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    #     span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    #     buffer = collections.deque(maxlen=span)
    #     for _ in range(span):
    #         buffer.append(data[self.data_index])
    #         self.data_index = (self.data_index + 1) % len(data)
    #     for i in range(batch_size // num_skips):
    #         target = skip_window  # target label at the center of the buffer
    #         targets_to_avoid = [skip_window]
    #         for j in range(num_skips):
    #             while target in targets_to_avoid:
    #                 target = random.randint(0, span - 1)
    #             targets_to_avoid.append(target)
    #             batch[i * num_skips + j] = buffer[skip_window]
    #             labels[i * num_skips + j, 0] = buffer[target]
    #         buffer.append(data[self.data_index])
    #         self.data_index = (self.data_index + 1) % len(data)
    #     return batch, labels
