#!/usr/bin/env python3
'''svdfdbnhmksk'''
import transformers
from setup import load_pt2en


class Dataset:
    '''this is class'''
    def __init__(self):
        '''wdccjncdnkdkdj'''
        self.data_train = load_pt2en(split='train')
        self.data_valid = load_pt2en(split='validation')
        self.tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
                'neuralmind/bert-base-portuguese-cased')
        self.tokenizer_en = transformers.AutoTokenizer.from_pretrained(
                'bert-base-uncased')

    def tokenize_dataset(self, data):
        '''sjfjjdfjdkjehuytsyf'''

