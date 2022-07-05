import torch
from torch import nn
from torchcrf import CRF
from embedding import getEmbedding
class BiLSTMCRF(nn.Module):
    def __init__(self,hidden_dim,tag_num,embed_dim,emb_pretrain=False,dropout=0, nl02=False):
        super(BiLSTMCRF,self).__init__()
        self.hidden_dim=hidden_dim
        self.embedding=getEmbedding(dim=embed_dim,pretrain=emb_pretrain,nl02=nl02)
        self.lstm_layer=torch.nn.LSTM(input_size=embed_dim,hidden_size=hidden_dim // 2,bidirectional=True,batch_first=True)
        self.linear_layer=nn.Linear(hidden_dim,tag_num)
        self.crf=CRF(tag_num,batch_first=True)
        self.dpout=torch.nn.Dropout(p=dropout)

    def lstm_forward(self,sentence):
        lstm_input=self.embedding(sentence)
        lstm_input=self.dpout(lstm_input)
        lstm_out,hidden_n=self.lstm_layer(lstm_input)
        return lstm_out

    def model_loss(self,lstm_tag, tags, mask):
        return - self.crf(lstm_tag, tags, mask=mask,reduction='mean')

    def best_seq(self,lstm_tag, mask):
        return self.crf.decode(lstm_tag, mask=mask)

    def forward(self,sentence):
        lstm_out=self.lstm_forward(sentence)
        lstm_tag=self.linear_layer(lstm_out)
        return lstm_tag
