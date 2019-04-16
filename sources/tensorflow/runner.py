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
        # Commented out for nlp fr now
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

                # Evaluate
                if global_step % 10000 == 0:
                    reverse_dictionary = data_loader.dataset.reverse_dictionary
                    model.evaluate(sess=sess, reverse_dictionary=reverse_dictionary)

            if early_stop_count > 2:
                learning_rate = learning_rate * 0.90

            if early_stop_count > 5:
                print('Early stopped !')
                break

