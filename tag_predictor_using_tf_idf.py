"""Author - Abhijith Shreesh"""

import pandas as pd
import os
from numpy import nan, array
import math
from collections import Counter
#from scipy.spatial.distance import cosine
import operator
from scipy.spatial import distance


class TagPredictionUsingTFIDF(object):

    def __init__(self):
        self.total_data_df = pd.read_csv(os.path.join("data", "cleaned_data.csv"), encoding = "ISO-8859-1")
        self.data_df = self.total_data_df[~self.total_data_df.Tags.isnull()]
        #self.total_tag_list = self.get_tag_list()
        self.total_word_list = self.get_word_list()
        #self.tag_to_index_dict = {self.total_tag_list[i]: i for i in range(len(self.total_tag_list))}
        self.word_to_index_dict = {self.total_word_list[i]: i for i in range(len(self.total_word_list))}
       # self.data_df['tag_vector'] = pd.Series([[0] * len(self.total_tag_list) for each in self.data_df.index], index=self.data_df.index)
        self.data_df['TF_IDF'] = pd.Series([[0] * len(self.total_word_list) for each in self.data_df.index], index=self.data_df.index)
        self.total_records = len(self.data_df .index)
        self.idf_dict = self.assign_idf_weight(self.data_df.stemmed_words)

    def get_tag_list(self):
        tag_set = set()
        for tags in self.data_df.Tags:
            if tags is not nan:
                tag_set.update(tags.split(','))
        return sorted(list(tag_set))

    def get_word_list(self):
        word_set = set()
        for words in self.data_df.stemmed_words:
            if words is not nan:
                word_set.update(words.split(' '))
        return sorted(list(word_set))

    def assign_idf_weight(self, data_series):
        idf_counter = {word: 0 for word in self.total_word_list}
        for word_list in data_series:
            for word in word_list.split(' '):
                idf_counter[word] += 1
        for word, count in list(idf_counter.items()):
            if count:
                idf_counter[word] = math.log(len(data_series.index)/count, 2)
        return idf_counter

    def assign_tf_weight(self, word_list):
        counter = Counter()
        for each in word_list:
            counter[each] += 1
        total = sum(counter.values())
        for each in counter:
            counter[each] = (counter[each]/total)
        return dict(counter)

    def assign_tf_idf_weight(self, tf_dict, tf_idf):
        for key, value in tf_dict.items():
            tf_idf[self.word_to_index_dict[key]] = value*self.idf_dict[key]
        return tf_idf

    def compute_tf_idf(self):
        self.data_df['TF'] = pd.Series([self.assign_tf_weight(words.split(' ')) for words
                                         in self.data_df.stemmed_words], index=self.data_df.index)
        self.data_df['TF_IDF'] = pd.Series([self.assign_tf_idf_weight(tf_dict, tf_idf) for
                                            tf_dict, tf_idf
                                            in zip(self.data_df.TF, self.data_df.TF_IDF)],
                                           index=self.data_df.index)

    def assign_tags(self, tf_idf_test, train_df, k):
        train_df = train_df.reset_index()
        #distace_dict = {index: cosine(tf_idf_test, tf_idf_train) for index, tf_idf_train in
        #                zip(train_df.index, train_df.TF_IDF)}
        #sorted_dict = sorted(distace_dict.items(), key=operator.itemgetter(1), reverse=False)
        test_record = array([tf_idf_test, tf_idf_test])
        res_dist = distance.cdist(train_df.TF_IDF.tolist(), test_record, metric='cosine')
        distance_list = [(i, res_dist[i][0]) for i in range(len(res_dist))]
        distance_list.sort(key=lambda tup: tup[1])
        index_list = [distance_list[i][0] for i in range(k)]
        #index_list = [each[0] for each in sorted_dict[0:k]]
        tags = train_df.iloc[index_list, 13].str.cat(sep=',')
        tf_of_tags = self.assign_tf_weight(tags.split(','))
        tags_tuple = sorted(tf_of_tags.items(), key=operator.itemgetter(1), reverse=True)
        return [each[0] for each in tags_tuple[0:10]]



    def predict_tags(self, k=10):
        train_df = self.data_df.tail(int(self.total_records * .67))
        test_df = self.data_df.head(int(self.total_records*.23))
        test_df['New_Tags'] = pd.Series([','.join(self.assign_tags(tf_idf, train_df, k)) for tf_idf in test_df.TF_IDF], index=test_df.index)
        test_df[['Body', 'body_without_stop_words', 'stemmed_words', 'Tags', 'New_Tags']].to_csv(os.path.join("data", "tf_idf_test.csv"))


if __name__ == "__main__":
    predictor = TagPredictionUsingTFIDF()
    predictor.compute_tf_idf()
    predictor.predict_tags()

