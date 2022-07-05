import os
import torch
from torch.utils.data import Dataset,DataLoader
vocabs = {'<unk>':0,'<pad>':1}
labelsmap = {'O':0,'B-LOC':1,'I-LOC':2,'B-PER':3,'I-PER':4,'B-ORG':5,'I-ORG':6,"B-MISC":7,"I-MISC":8, 'PAD':-1}
labelsmap_inverse = ['O','B-LOC','I-LOC','B-PER','I-PER','B-ORG','I-ORG',"B-MISC","I-MISC", 'PAD']
class CoNLLset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None, train=True, lower_case=True, nl02=False):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.train = train
        self.read_data_file(lower_case, nl02)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0],self.data[idx][1]

    def read_data_file(self,lower_case,nl02):
        if self.train:
            if nl02:
                sub_dir = 'esp.train'
            else:
                sub_dir = 'train.txt'
        else:
            if nl02:
                sub_dir = 'esp.testa'
            else:
                sub_dir = 'valid.txt'
        with open(os.path.join(self.data_dir,sub_dir),'r') as f:
            sentence = []
            labelseq = []
            for line in f:
                if len(line) > 1 and line[:10] != '-DOCSTART-':
                    temp = line.strip().split(' ')
                    labelseq.append(labelsmap[temp[-1]])
                    word_idx=0
                    if lower_case:
                        temp[0]=temp[0].lower()# because glove is lower case
                    if temp[0] not in vocabs:
                        if self.train:
                            word_idx=len(vocabs)
                            vocabs[temp[0]] = word_idx
                        else:
                            word_idx=0
                    else:
                        word_idx=vocabs[temp[0]]
                    sentence.append(word_idx)
                else:
                    if len(sentence) > 0:
                        self.data.append((sentence,labelseq))
                    sentence = []
                    labelseq = []


def conll_collate_fn(data):
    maxseqlen = len(max(data,key=lambda x:len(x[0]))[0])#get longest sentence length
    x = []
    y = []
    MASK = [] 
    for entry in data:
        #do padding
        x.append(entry[0] + [vocabs['<pad>'] for i in range(maxseqlen - len(entry[0]))])
        y.append(entry[1] + [labelsmap['PAD'] for i in range(maxseqlen - len(entry[1]))])
        MASK.append([1 for i in range(len(entry[0]))]+[0 for i in range(maxseqlen - len(entry[0]))])
    return torch.tensor(x),torch.tensor(y),torch.tensor(MASK,dtype=torch.uint8)
