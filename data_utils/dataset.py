from __future__ import absolute_import, division, print_function, unicode_literals

import codecs
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pprint import pprint
from typing import Tuple, Callable, List # https://m.blog.naver.com/PostView.nhn?blogId=passion053&logNo=221070020739&proxyReferer=https%3A%2F%2Fwww.google.com%2F
import pickle
import json
from tqdm import tqdm
from collections import OrderedDict
import re
from gluonnlp.data import SentencepieceTokenizer, SentencepieceDetokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from data_utils.vocab_tokenizer import Vocabulary, Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from pathlib import Path
class DocumentDataset(Dataset):
    def __init__(self, train_data_dir: str, model_dir=Path('data_in')) -> None:
        """
        :param train_data_in:
        :param transform_fn:
        """
        self.model_dir = model_dir

        list_of_doc, list_of_tag = self.load_data(train_data_dir=train_data_dir)
        # input(list_of_doc)
        self._corpus = list_of_doc
        self._label = list_of_tag
        self.creat_tag_index(list_of_tag)


    def set_transform_fn(self, transform_source_fn, transform_target_fn):

        self._transform_source_fn = transform_source_fn
        self._transform_target_fn = transform_target_fn



    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        token_ids_with_cls_sep, tokens = self._transform_source_fn(self._corpus[idx].lower())

        list_of_tag_ids = self._transform_target_fn(self._label[idx], tokens)

        x_input = torch.tensor(token_ids_with_cls_sep).long()

        token_type_ids = torch.tensor(len(x_input[0]) * [0])
        label = torch.tensor(list_of_tag_ids).long()

        return x_input[0], token_type_ids, label

    def creat_tag_index(self,  list_of_tag):
        """ if you want to build new json file, you should delete old version. """

        if not os.path.exists(self.model_dir / "tag_to_index.json"):

            tag_to_index = {"[CLS]":0, "[SEP]":1, "[PAD]":2, "[MASK]":3, }
            for tag in list_of_tag:
                if not tag in tag_to_index:
                    tag_to_index[tag] = len(tag_to_index)


            # save ner dict in data_in directory
            with open(self.model_dir / 'tag_to_index.json', 'w', encoding='utf-8') as io:
                json.dump(tag_to_index, io, ensure_ascii=False, indent=4)
            self.tag_to_index = tag_to_index
        else:
            self.set_tag_dict()

    def set_tag_dict(self):
        with open(self.model_dir / "tag_to_index.json", 'rb') as f:
            self.tag_to_index = json.load(f)

    ### 여기서 path를 바꿔서 트레인 데이터 변경
    def load_data(self, train_data_dir):
        # if you want train
        file_name = "train_data.txt"
        # if you want test
        # file_name = "test.txt"

        file_path = train_data_dir / file_name
        list_of_doc, list_of_tag = self.load_data_from_txt(file_full_name = file_path)
        assert len(list_of_tag) == len(list_of_doc)
        return list_of_doc, list_of_tag


    def load_data_from_txt(self, file_full_name):
        list_of_doc = []
        list_of_tag = []
        with codecs.open(file_full_name, "r", "utf-8") as io:
            lines = io.readlines()

            for line in lines:

                list_of_tag.append(line.split("\t")[0])
                list_of_doc.append(line.split("\t")[1].strip("\r\n"))

        return list_of_doc, list_of_tag




class DocumentFormatter():
    """ NER formatter class """
    def __init__(self, vocab=None, tokenizer=None, maxlen=30, model_dir=Path('data_in')):

        if vocab is None or tokenizer is None:
            tok_path = get_tokenizer()
            self.ptr_tokenizer = SentencepieceTokenizer(tok_path)
            self.ptr_detokenizer = SentencepieceDetokenizer(tok_path)
            _, vocab_of_gluonnlp = get_pytorch_kobert_model()
            token2idx = vocab_of_gluonnlp.token_to_idx
            self.vocab = Vocabulary(token_to_idx=token2idx)
            self.tokenizer = Tokenizer(vocab=self.vocab, split_fn=self.ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=maxlen)
        else:
            self.vocab = vocab
            self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.model_dir = model_dir

    def transform_source_fn(self, text):
        # text = "첫 회를 시작으로 13일까지 4일간 총 4회에 걸쳐 매 회 2편씩 총 8편이 공개될 예정이다."

        tokens = self.tokenizer.split(text)
        token_ids_with_cls_sep = self.tokenizer.list_of_string_to_arr_of_cls_sep_pad_token_ids([text])

        return token_ids_with_cls_sep, tokens




    def transform_target_fn(self, tag, tokens):
        """
        인풋 토큰에 대응되는 index가 토큰화된 엔티티의 index 범위 내에 있는지 체크해서 list_of_ner_ids를 생성함
        이를 위해서 B 태그가 시작되었는지 아닌지도 체크해야함
        매칭하면 entity index를 증가시켜서 다음 엔티티에 대해서도 검사함
        :param label_text:
        :param tokens:
        :param prefix_sum_of_token_start_index:
        :return:
        """

        list_of_ids = []
        with open(self.model_dir / "tag_to_index.json", 'rb') as f:
            self.tag_to_index = json.load(f)
        list_of_ids = self.tag_to_index[tag]
        return list_of_ids
