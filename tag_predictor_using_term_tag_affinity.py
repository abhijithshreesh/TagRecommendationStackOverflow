"""Author - Abhijith Shreesh"""

import pandas as pd
import os
from numpy import nan, array
import operator


class TagPredictionUsingTermTagAffinity(object):

    def __init__(self):
        self.total_data_df = pd.read_csv(os.path.join("data", "cleaned_data.csv"))
        self.data_df = self.total_data_df[~self.total_data_df.Tags.isnull()]
        self.total_records = len(self.data_df.index)
        self.train_df = self.data_df.tail(int(self.total_records * .67))
        self.test_df = self.data_df.head(int(self.total_records * .23))
        self.total_tag_list = self.get_tag_list()
        self.total_word_list = self.get_word_list()
        self.tag_to_index_dict = {self.total_tag_list[i]: i for i in range(len(self.total_tag_list))}
        self.word_to_index_dict = {self.total_word_list[i]: i for i in range(len(self.total_word_list))}
        self.word_to_tag_dict = {word: {tag: 0 for tag in self.total_tag_list} for word in self.total_word_list}
        self.count = 0

    def get_tag_list(self):
        tag_set = set()
        for tags in self.train_df.Tags:
            if tags is not nan:
                tag_set.update(tags.split(','))
        return sorted(list(tag_set))

    def get_word_list(self):
        word_set = set()
        for words in self.train_df.stemmed_words:
            if words is not nan:
                word_set.update(words.split(' '))
        return sorted(list(word_set))

    def update_word_tag_count_dict(self):
        for words, tags in zip(self.train_df.stemmed_words, self.train_df.Tags):
            if words is not nan:
                word_list = words.split(' ')
                for word in word_list:
                    for tag in tags.split(','):
                        self.word_to_tag_dict[word][tag] += 1

    def get_new_tags(self, words):
        self.count += 1
        tags = dict()
        if words is not nan:
            for word in words.split(' '):
                tags_dict = self.word_to_tag_dict.get(word)
                if tags_dict:
                    for key, value in tags_dict.items():
                        if key in tags.keys():
                            tags[key] += value
                        else:
                            tags[key] = value
        sorted_tags = sorted(tags.items(), key=operator.itemgetter(1), reverse=True)
        tag_list = [sorted_tags[i][0] for i in range(len(sorted_tags))]
        return ','.join(tag_list[0:15])

    def test_term_tag_affinity_model(self):
        self.test_df['New_Tags'] = pd.Series([self.get_new_tags(words) for words in self.test_df.stemmed_words], index=self.test_df.index)
        self.test_df[['Body', 'body_without_stop_words', 'stemmed_words', 'Tags', 'New_Tags']].to_csv(os.path.join("data", "term_tag_affinity_test.csv"))


if __name__ == "__main__":
    predictor = TagPredictionUsingTermTagAffinity()
    predictor.update_word_tag_count_dict()
    predictor.test_term_tag_affinity_model()
    print(predictor.count)



