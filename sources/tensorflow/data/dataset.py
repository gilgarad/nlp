import random
from .preprocess import PreProcess

import os
import zipfile
from six.moves import urllib


class DataSet:
    def __init__(self, data_path, hparams):
        """

        :param data_path:
        :param data_status: data's status 0 - complete raw data, 1 - sequenced data, 2 - sequenced with label data
        :param label_price: label classifier or actual price finder?
        :param mode: 0 Many to many
        :param sequence_length:
        :param label_term:
        """
        # Current data format
        # date, final_price, compare_to_prior, start_price, highest_price, lowest_price, num_of_traded

        self.hparams = hparams
        vocabulary_size = 50000
        words = self.samples()
        data, count, dictionary, reverse_dictionary = PreProcess.build_dictionary(words, vocabulary_size)

        self.data = data
        self.data_index = 0
        self.reverse_dictionary = reverse_dictionary

        print('Most common words (+UNK)', count[:5])
        print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

        batch, labels, self.data_index = PreProcess.generate_batch(batch_size=8, num_skips=2, skip_window=1,
                                                                   data=data, data_index=self.data_index)
        for i in range(8):
            print(batch[i], reverse_dictionary[batch[i]],
                  '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
        print('Input X')
        print(batch)
        print('Output Y')
        print(labels)

        # print('Number of data: %i, %i' % (len(self.sequence_data), len(self.sequence_data)))

    # def get_label_dict(self):
    #     label_dict = dict()
    #     label_range = [-30, 30]  # min max
    #     num_labels = int((label_range[1] - label_range[0]) / self.label_term)
    #     prev = label_range[0]
    #     for i in range(num_labels):
    #         label_dict[i] = [prev, prev + self.label_term]
    #         prev += self.label_term
    #     label_dict[num_labels - 1][1] = label_range[1] + 1
    #
    #     print('Label Range:', label_dict)
    #
    #     return label_dict

    def samples(self):
        # Step 1: 필요한 데이터를 다운로드한다.
        url = 'http://mattmahoney.net/dc/'
        filename = 'text8.zip'
        expected_bytes = 31344016


        """파일이 존재하지 않으면 다운로드하고 사이즈가 적절한지 체크한다."""
        if not os.path.exists(filename):
            filename, _ = urllib.request.urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        print(statinfo.st_size, expected_bytes)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')

        """zip파일 압축을 해제하고 단어들의 리스트를 읽는다."""
        with zipfile.ZipFile(filename) as f:
            words = f.read(f.namelist()[0]).split()

        words = [w.decode("utf-8") for w in words]

        print('Data size', len(words))

        return words

    def __len__(self):
        # return len(self.sequence_data)
        return len(self.data)

    def __getitem__(self, idx):

        queries, labels, self.data_index = PreProcess.generate_batch(batch_size=self.hparams.batch_size,
                                                                     num_skips=self.hparams.num_skips,
                                                                     skip_window=self.hparams.skip_window,
                                                                     data=self.data,
                                                                     data_index=self.data_index)

        return queries, labels

    def reshuffle(self):
        pass
        # combined = list(zip(self.sequence_data, self.end_data))
        # random.shuffle(combined)
        # self.sequence_data, self.end_data = zip(*combined)

