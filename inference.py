from __future__ import absolute_import, division, print_function, unicode_literals
import json
import pickle
import argparse
import torch
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from model.net import KobertSequenceFeatureExtractor
from gluonnlp.data import SentencepieceTokenizer
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from pathlib import Path


def read_score_comments(fname):
    score5 = []
    score4 = []
    score2 = []
    score1 = []
    scores = [score1, score2, score4, score5]
    with open(fname, "r", encoding="UTF-8") as f:

        for line in f.readlines():
            line = line.split("\t")
            score = line[0]
            comment = line[1].strip()

            if score == "5":
                score5.append(("5", comment))
            elif score == "4":
                score4.append(("4", comment))
            elif score == "2":
                score2.append(("2", comment))
            else:
                score1.append(("1", comment))
    return scores

def tuple_to_list(scores):
    list_comments, list_labels= [], []
    for score in scores:
        tmp1 = []
        tmp2 = []
        for (label, comment) in score:
            tmp1.append(comment)
            tmp2.append(label)
        list_labels.append(tmp1)
        list_comments.append(tmp2)

    return list_comments, list_labels

def check_acc(actual, predict):
    size, N, PF,  FP, FN = [0,0,0,0], [0,0,0,0,0],[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]
    for i, score in enumerate(actual):
        tmp = 0
        for j, value in enumerate(score):
            if value == predict[i][j]:
                PF[int(value)-1]+=1
                # tmp +=1
            else:
                FP[int(predict[i][j])-1] += 1
                FN[int(value)-1] +=1
            N[int(predict[i][j])-1] += 1
        # PF.append(tmp)
        size[i] = len(score)
    return size, PF, N, FP, FN

def main(parser):

    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    model_config = Config(json_path=model_dir / 'config.json')

    # Vocab & Tokenizer
    # tok_path = get_tokenizer() # ./tokenizer_78b3253a26.model
    tok_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    # load vocab & tokenizer
    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

    # load ner_to_index.json
    with open(model_dir / "tag_to_index.json", 'rb') as f:
        tag_to_index = json.load(f)
        index_to_tag = {v: k for k, v in tag_to_index.items()}

    # Model
    model = KobertSequenceFeatureExtractor(config=model_config, num_classes=len(tag_to_index))


    # load
    model_dict = model.state_dict()
    checkpoint = torch.load("./experiments/base_model/best-epoch-28-step-139900-acc-1.000.bin", map_location=torch.device('cpu'))


    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    model.load_state_dict(convert_keys)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    decoder_from_res = DecoderDoc(tokenizer=tokenizer, index_to_tag=index_to_tag)

    # n_gpu = torch.cuda.device_count()
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)
    model.to(device)
    list_of_tag, list_of_comment = tuple_to_list(read_score_comments("data_in/test_data.txt"))
    list_of_pred_tag = []
    for i in range(4):
        tmp = []
        print("-"*4, i,"ing -"*4)
        for idx, input_txt in enumerate(list_of_comment[i]):
            # print(input_txt)
            list_of_input_ids = tokenizer.list_of_string_to_arr_of_cls_sep_pad_token_ids([input_txt])
            x_input = torch.tensor(list_of_input_ids).long()

            y_pred = model(x_input.to(device))
            list_of_pred_ids = y_pred.max(dim=-1)[1].tolist()

            pred_tag = decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)
            tmp.append(pred_tag)
        list_of_pred_tag.append(tmp)

    class_size, PF, N, FP, FN = check_acc(list_of_tag, list_of_pred_tag)
    tmp, acc = [0, 1, 3, 4], [0, 0, 0, 0]
    print(check_acc(list_of_tag, list_of_pred_tag))
    for i in range(4):
        acc[i] = PF[tmp[i]] / class_size[i]
    print(tag_to_index)
    for i in range(4):
        # print("-"*8, tag_to_index[tmp[i]],"-"*8 )
        print("ACC : {}, N : {}, PF : {}, FP : {}, FN : {}".format(acc[i], N[tmp[i]], PF[tmp[i]], FP[tmp[i]], FN[tmp[i]]))

class DecoderDoc():
    def __init__(self, tokenizer, index_to_tag):
        self.tokenizer = tokenizer
        self.index_to_tag = index_to_tag

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        pred_tag = self.index_to_tag[list_of_pred_ids[0]]


        return pred_tag
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data_in', help="Directory containing config.json of data")
    parser.add_argument('--model_dir', default='./experiments/base_model', help="Directory containing config.json of model")
   

    main(parser)