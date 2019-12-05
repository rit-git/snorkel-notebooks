import spacy
from nltk.tokenize import word_tokenize

class Candidate:
    def __init__(self, df_row):
        self.__dict__.update(df_row.to_dict())
        self.tokens = word_tokenize(self.text)
        self.doc_id = df_row.id

    def __repr__(self):
        return str(vars(self))

    def __getitem__(self, index):
        return objectview({
            'doc_id': self.doc_id,
            'entity': None, 
            'text': self.text, 
            'tokens': self.tokens,
            'words': self.tokens,
            'char_offsets': [], 
            'pos_tags': [], 
            'ner_tags': [], 
            'entity_types': []
        })



class objectview(object):
    def __init__(self, d):
        self.__dict__ = d