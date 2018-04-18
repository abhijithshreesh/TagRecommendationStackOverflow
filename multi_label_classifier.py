"""Author - Abhijith Shreesh"""

import pandas as pd
import os
from numpy import nan, array
import operator
from sklearn.naive_bayes import GaussianNB

class MultiLabelClassifier(object):

    def __init__(self):
        self.total_data_df = pd.read_csv(os.path.join("data", "cleaned_data.csv"))
        self.data_df = self.total_data_df[~self.total_data_df.Tags.isnull()]
        self.total_records = len(self.data_df.index)
        self.train_df = self.data_df.tail(int(self.total_records * .67))
        self.test_df = self.data_df.head(int(self.total_records * .23))
        self.total_tag_list = self.get_tag_list()
        self.total_word_list = self.get_word_list()
        self.modified_train_df = pd.DataFrame()
        self.modified_test_df = pd.DataFrame()
        self.classifier = GaussianNB()

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

    def setup_data_frame(self):
        for each in self.total_word_list:
            self.modified_train_df[each] = pd.Series([1 if each in words.split(' ') else 0 for words in self.train_df.stemmed_words], index=self.train_df.index)
            self.modified_test_df[each] = pd.Series([1 if each in words.split(' ') else 0 for words in self.test_df.stemmed_words], index=self.test_df.index)
        for tag in self.total_tag_list:
            self.modified_train_df[tag] = pd.Series([1 if tag in tags.split(',') else 0 for tags in self.train_df.Tags], index=self.train_df.index)
        return self.modified_train_df

    def multi_label_naive_bayes_classifier(self):
        test_rows = self.modified_test_df.values
        self.modified_test_df['predicted_labels'] = pd.Series(['' for each in self.modified_test_df.index], index=self.modified_test_df.index)
        for tag in self.total_tag_list:
            self.classifier.fit(self.modified_train_df[self.total_word_list].values, self.modified_train_df[tag].tolist())
            self.modified_test_df[tag] = pd.Series(self.classifier.predict(test_rows), index=self.modified_test_df.index)
            self.modified_test_df['predicted_labels'] = pd.Series([each + ',' + tag if value==1 else each for
                                                                   each,value in zip(self.modified_test_df.predicted_labels,
                                                                    self.modified_test_df.tag)], index=self.modified_test_df.index)

if __name__ == "__main__":
    predictor = MultiLabelClassifier()
    df = predictor.setup_data_frame()
    predictor.multi_label_naive_bayes_classifier()
    test_df = predictor.test_df.join(predictor.modified_test_df, how='inner')
    a=1