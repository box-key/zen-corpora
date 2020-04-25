import torch
import torch.nn as nn


class Encoder(nn.Module):


    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 dropout,
                 device):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(
                    input_size=emb_dim,
                    hidden_size=hid_dim,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=False
                   )
        self.dropout = nn.Dropout(dropout)
        self.device = device


    def forward(self, keywords, keywords_len):
        # keywords = [keywords_len, batch_size]
        embedded = self.embedding(keywords)
        # embedded = [keywords_len, batch_size, emb_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, keywords_len)
        _, hidden = self.rnn(packed_embedded)
        # hidden = [n_layers, batch_size, hid_dim]
        return hidden


class Decoder(nn.Module):


    def __init__(self,
                 input_dim,
                 output_dim,
                 emb_dim,
                 hid_dim,
                 n_layers,
                 dropout,
                 device):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        # self.bert = bert
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(
                    input_size=emb_dim,
                    hidden_size=hid_dim,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=False
                  )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device


    def forward(self, input, hidden):
        # input = [batch_size]
        # hidden = [batch_size, hid_dim]
        input = input.unsqueeze(0)
        # input = [1, batch_size]
        embedded = self.embedding(input)
        # embedded = [1, batch_size, emb_dim]
        output, hidden = self.rnn(embedded, hidden)
        # output = [1, batch_size, hid_dim]
        # hidden = [4, batch_size, hid_dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden


class Seq2Seq(nn.Module):


    def __init__(self,
                 enc_input_dim,
                 dec_input_dim,
                 emb_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 output_dim,
                 n_layers,
                 dropout,
                 device):
        super().__init__()
        self.encoder = Encoder(
                           input_dim=enc_input_dim,
                           emb_dim=emb_dim,
                           hid_dim=enc_hid_dim,
                           n_layers=n_layers,
                           dropout=dropout,
                           device=device
                       )
        self.decoder = Decoder(
                           input_dim=dec_input_dim,
                           output_dim=output_dim,
                           emb_dim=emb_dim,
                           hid_dim=dec_hid_dim,
                           n_layers=n_layers,
                           dropout=dropout,
                           device=device
                       )
        self.n_layers = n_layers
        self.device = device


    def forward(self, keywords, keywords_len, trg):
        # keywords_tuple = [keywords_len, batch_size]
        # keywords_len = [batch_size]
        # trg = [trg_len, batch_size]
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        prediction_seq = torch.zeros(trg_len, batch_size, self.decoder.output_dim).to(self.device)
        # prediction_seq = [trg_len, batch_size, output_dim]
        hidden = self.encoder(keywords, keywords_len)
        # hidden = [n_layers, batch_size, enc_hid_dim]
        input = trg[0,:]
        # input = [batch_size]
        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input, hidden)
            # hidden = [n_layers, batch_size, dec_hid_dim]
            # prediction = [batch_size, output_dim]
            prediction_seq[t] = prediction
            # get the highest predicted token from prediction
            top_token =prediction.argmax(1)
            input = top_token
            # input = [batch_size]
        return prediction_seq
