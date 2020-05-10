import os
import torch
import dill
import csv

from test.seq2seq_multilayer_gru_with_pad import Seq2Seq


PATH_MODEL = os.path.join('test', 'data', 'seq2seq-multilayer-gru.pt')
PATH_SRC_VOCAB = os.path.join('test', 'data', 'MASKED_TEXT.Field')
PATH_TRG_VOCAB = os.path.join('test', 'data', 'TARGET_TEXT.Field')
PATH_CORPUS_SAMPLE = os.path.join('test', 'data', 'space_sample.csv')
PATH_CORPUS_MASTER = os.path.join('test', 'data', 'search_space.csv')

EMB_DIM = 256
N_LAYER = 4
HID_DIM = 1024
DROPOUT = 0.3


class DataLoader():

    def __init__(self, small_corpus=True):
        self.input_field = self._load_dill(PATH_SRC_VOCAB)
        self.output_field = self._load_dill(PATH_TRG_VOCAB)
        self.device = torch.device('cpu')
        self.model = Seq2Seq(
              device=self.device,
              enc_input_dim=len(self.input_field.vocab),
              dec_input_dim=len(self.output_field.vocab),
              output_dim=len(self.output_field.vocab),
              emb_dim=EMB_DIM,
              enc_hid_dim=HID_DIM,
              dec_hid_dim=HID_DIM,
              n_layers=N_LAYER,
              dropout=DROPOUT
        ).to(self.device)
        self._load_model(PATH_MODEL)
        path = PATH_CORPUS_SAMPLE if small_corpus else PATH_CORPUS_MASTER
        self.corpus = self._load_corpus(path)

    def _load_dill(self, path):
        with open(path, "rb") as f:
            obj = dill.load(f)
        return obj

    def _load_model(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )

    def _load_corpus(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            file = csv.reader(f, delimiter=',')
            corpus = [row[0].split() for row in file]
        return corpus[1:]
