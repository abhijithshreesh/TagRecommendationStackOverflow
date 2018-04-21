"""Author - Abhijith Shreesh"""
import pandas as pd
import os
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from re import sub
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re
from numpy import nan

class XML2DataFrame(object):

    def __init__(self, xml_file):
        self.root = ET.parse(xml_file).getroot()

    def parse_root(self, root):
        """Return a list of dictionaries from the text
         and attributes of the children under this XML root."""
        return [self.parse_element(child) for child in iter(root)]

    def parse_element(self, element, parsed=None):
        """ Collect {key:attribute} and {tag:text} from thie XML
         element and all its children into a single dictionary of strings."""
        if parsed is None:
            parsed = dict()

        for key in element.keys():
            if key not in parsed:
                parsed[key] = element.attrib.get(key)
            else:
                raise ValueError('duplicate attribute {0} at element {1}'.format(key, element.getroottree().getpath(element)))


        for child in list(element):
            self.parse_element(child, parsed)

        return parsed

    def process_data(self):
        """ Initiate the root XML, parse it, and return a dataframe"""
        structure_data = self.parse_root(self.root)
        return pd.DataFrame(structure_data)

class DataCleaner(object):
    def __init__(self, raw_data_df):
        self.raw_data_df = raw_data_df
        self.cache_stop_words = stopwords.words('english')
        self.required_columns = ["AcceptedAnswerId","AnswerCount","Body","CommentCount","CreationDate","FavoriteCount",
                                 "Id","LastActivityDate","LastEditDate","ParentId","PostTypeId","Score","Tags","Title",
                                 "ViewCount", "body_without_stop_words"]
        self.ps = SnowballStemmer("english")

    def clean_html_tags(self, raw_string):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_string)
        return cleantext

    def remove_stop_words_from_string(self, string):
        modified_string = sub('[^ a-zA-Z]', ' ', string)
        modified_string = ' '.join([word for word in modified_string.lower().split(' ') if word not in self.cache_stop_words])
        return modified_string

    def stem_data(self, string):
        words = word_tokenize(string)
        stem_list = [self.ps.stem(w) for w in words]
        if "" in stem_list:
            stem_list.remove("")
        return stem_list

    def extract_tags(self, tag_string):
        tag_list =[]
        if tag_string and tag_string is not nan:
            tag_string = tag_string.replace("<", "")
            tag_string = tag_string.replace(">", " ")
            tag_list = tag_string.split(" ")
            tag_list = [each.strip(' ') for each in tag_list]
            tag_list.remove('')
        return tag_list

    def clean_data(self):
        self.raw_data_df['cleaned_body'] = pd.Series([self.clean_html_tags(each) for each in self.raw_data_df.Body], index = self.raw_data_df.index)
        self.raw_data_df['body_without_stop_words'] = pd.Series([self.remove_stop_words_from_string(each)
                                                       for each in self.raw_data_df.cleaned_body],
                                                       index=self.raw_data_df.index)

        cleaned_df = self.raw_data_df[self.required_columns]
        cleaned_df['stemmed_words'] = pd.Series([self.stem_data(each)
                                                       for each in self.raw_data_df.body_without_stop_words],
                                                       index = self.raw_data_df.index)

        cleaned_df['stemmed_words'] = pd.Series([self.remove_stop_words_from_string(' '.join(each))
                                                                 for each in cleaned_df.stemmed_words],
                                                                 index=cleaned_df.index)
        cleaned_df['Tags'] = pd.Series([','.join(self.extract_tags(tag_string)) for tag_string in cleaned_df.Tags], index=cleaned_df.index)

        return cleaned_df

def trigger_data_extracttor():
    xml_file = os.path.join("data", "Posts.xml")
    xml2df = XML2DataFrame(xml_file)
    xml_dataframe = xml2df.process_data()
    xml_dataframe.to_csv(os.path.join("data", "parsed_data.csv"), index=False)
    data_cleaner = DataCleaner(xml_dataframe)
    cleaned_data = data_cleaner.clean_data()
    cleaned_data.to_csv(os.path.join("data", "cleaned_data.csv"), index=False)

if __name__ == "__main__":
    trigger_data_extracttor()
