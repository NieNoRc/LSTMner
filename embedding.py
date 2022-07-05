import os
import torch
from torch import nn
from dataproc import CoNLLset,vocabs

def getEmbedding(dim,pretrain=True,nl02=False):
    shape = (len(vocabs), dim)
    if pretrain:
        lang='en'
        if nl02:
            lang='esp'
        if os.path.exists('embd_weight/glove'+str(dim)+'dembd_'+lang+'.pt'): 
            embTensor=torch.load('embd_weight/glove'+str(dim)+'dembd_'+lang+'.pt')
            embedding= nn.Embedding.from_pretrained(embTensor, freeze=True, padding_idx=vocabs['<pad>'])
        else:
            embTensor = torch.randn(shape)
            read_from_glove('glove/glove.6B.'+str(dim)+'d.txt',embTensor)
            if not os.path.exists('embd_weight'):
                os.mkdir('embd_weight')
            lang='en'
            if nl02:
                lang='esp'
            torch.save(embTensor,'embd_weight/glove'+str(dim)+'dembd_'+lang+'.pt')
            embedding=nn.Embedding.from_pretrained(embTensor, freeze=True, padding_idx=vocabs['<pad>'])
    else:
        embedding=torch.nn.Embedding(len(vocabs), dim, padding_idx=vocabs['<pad>'])
        embedding.weight.requires_grad = False
    return embedding

def read_from_glove(dir,embTensor):
    hit=0
    with open(dir,'r',encoding='utf-8') as f:
        dim=embTensor.shape[1]
        for line in f:
            temp=line.split(' ')
            if temp[0] in vocabs:
                hit+=1
                word_idx=vocabs[temp[0]]
                for i in range(0,dim):
                    embTensor[word_idx][i]=float(temp[i+1])
    print('Pretrain Embedding hit: '+str(hit))
    return embTensor
