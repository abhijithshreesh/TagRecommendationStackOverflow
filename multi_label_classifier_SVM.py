
"""Author - Abhijith Shreesh"""

import pandas as pd
import os
from numpy import nan, array
import operator
from sklearn.naive_bayes import BernoulliNB
from skmultilearn.problem_transform import ClassifierChain
import scipy.sparse as sps
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm



class MultiLabelClassifier(object):

    def __init__(self):
        self.total_data_df = pd.read_csv(os.path.join("data", "cleaned_data.csv"), encoding = "ISO-8859-1")
        self.data_df = self.total_data_df[~self.total_data_df.Tags.isnull()]
        self.total_records = len(self.data_df.index)
        self.train_df = self.data_df.tail(int(self.total_records * .67))
        self.test_df = self.data_df.head(int(self.total_records * .23))
        self.total_tag_list = self.get_tag_list()
        self.total_word_list = self.get_word_list()
        self.modified_train_df = pd.DataFrame()
        self.modified_test_df = pd.DataFrame()
        self.classifier = BernoulliNB()
        self.classifier_multilabel = ClassifierChain(BernoulliNB())
        self.classifier_dt = DecisionTreeRegressor(max_depth=2000)
        self.classifier_random_forest = RandomForestRegressor(max_depth=100)
        self.classifier_svm = svm.SVC(kernel='polynomial')

        self.test_tags = pd.DataFrame()

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
            self.test_tags[tag] = pd.Series([1 if tag in tags.split(',') else 0 for tags in self.test_df.Tags], index=self.test_df.index)
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

    def multi_label_naive_bayes_classifier_sklearn(self):
        test_rows = self.modified_test_df.values
        self.classifier_multilabel.fit(self.modified_train_df[self.total_word_list].values,
                                       self.modified_train_df[self.total_tag_list])
        c = self.classifier_multilabel.predict(test_rows)
        
        print(c.shape)
        print(sps.csc_matrix(self.test_tags.values).shape)
        print(accuracy_score(sps.csc_matrix(self.test_tags.values),c))

    def multi_label_decision_tree_regressor(self):
        test_rows = self.modified_test_df.values
        self.classifier_dt.fit(self.modified_train_df[self.total_word_list].values,
                               self.modified_train_df[self.total_tag_list])
        predictions = self.classifier_dt.predict(test_rows)
        temp_df = pd.DataFrame(predictions, columns=self.total_tag_list)
        self.test_df['predicted_labels'] = pd.Series(['' for each in self.modified_test_df.index],
                                                     index=self.modified_test_df.index)
        for tag in self.total_tag_list:
            self.test_df['predicted_labels'] = pd.Series([each + ',' + tag if value == 1 else each for
                                                          each, value in zip(self.test_df.predicted_labels,
                                                                             temp_df[tag])], index=self.test_df.index)
        self.test_df[['stemmed_words', 'Tags', 'predicted_labels']].to_csv(os.path.join("data", "decision_tree_result.csv"),
                                                                           index=False)

    def multi_label_random_forest(self):
        test_rows = self.modified_test_df.values
        self.classifier_random_forest.fit(self.modified_train_df[self.total_word_list].values,
                               self.modified_train_df[self.total_tag_list])
        predictions = self.classifier_random_forest.predict(test_rows)
        temp_df = pd.DataFrame(predictions, columns=self.total_tag_list)
        self.test_df['predicted_labels'] = pd.Series(['' for each in self.modified_test_df.index],
                                                     index=self.modified_test_df.index)
        for tag in self.total_tag_list:
            self.test_df['predicted_labels'] = pd.Series([each + ',' + tag if value == 1 else each for
                                                          each, value in zip(self.test_df.predicted_labels,
                                                                             temp_df[tag])], index=self.test_df.index)
        self.test_df[['stemmed_words', 'Tags', 'predicted_labels']].to_csv(os.path.join("data", "random_forest_result.csv"),
                                                                           index=False)

    def multi_label_svm(self):
        test_rows = self.modified_test_df.values
        tags = array(self.modified_train_df[self.total_tag_list])
        tag_t = tags.transpose()
        temp_df = pd.DataFrame()
        for col in range(tags.shape[1]):
            self.classifier_svm.fit(self.modified_train_df[self.total_word_list].values,
                                          tags[:,col])
            predictions = self.classifier_svm.predict(test_rows)
            temp_df[self.total_tag_list[col]] = pd.Series(predictions)
        #temp_df = pd.DataFrame(predictions, columns=self.total_tag_list)
        self.test_df['predicted_labels'] = pd.Series(['' for each in self.modified_test_df.index],
                                                     index=self.modified_test_df.index)
        for tag in self.total_tag_list:
            self.test_df['predicted_labels'] = pd.Series([each + ',' + tag if value == 1 else each for
                                                          each, value in zip(self.test_df.predicted_labels,
                                                                             temp_df[tag])], index=self.test_df.index)
        self.test_df[['stemmed_words', 'Tags', 'predicted_labels']].to_csv(
            os.path.join("data", "polynomial_svm.csv"),
            index=False)

if __name__ == "__main__":
    predictor = MultiLabelClassifier()
    df = predictor.setup_data_frame()
    """Jagdeesh commented these 4 below lines"""
    # predictor.multi_label_naive_bayes_classifier_sklearn()
    # test_df = predictor.test_df.join(predictor.modified_test_df, how='inner')
    # test_df[[ 'stemmed_words', 'Tags', 'predicted_labels']].to_csv("naivebayes.csv")
    # a=1
    """Jagdeesh commented these 4 above lines"""
    # predictor.multi_label_decision_tree_regressor()
    # predictor.multi_label_random_forest()
    predictor.multi_label_svm()