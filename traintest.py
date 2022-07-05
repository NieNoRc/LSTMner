import torch
from torch import nn
from mainmodel import BiLSTMCRF
from dataproc import CoNLLset,conll_collate_fn,vocabs,labelsmap,labelsmap_inverse
from torch.utils.data import DataLoader
from timeit import default_timer
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
class ResultStat():
    def __init__(self):
        self.pred = []
        self.truth = []

    #seqeval accept original string tag, need retrieve the tag of tag_index 
    def accum(self, pre,truthT, MASKT):
        batch_size, max_seq_len = MASKT.shape
        for i in range(0,batch_size):
            #cut the pred and y by using MASK
            temp_seq_len = MASKT[i].sum().item()
            temp_pred = []
            temp_truth = []
            for j in range(0,temp_seq_len):
                temp_pred.append(labelsmap_inverse[pre[i][j]])
                temp_truth.append(labelsmap_inverse[truthT[i][j].item()])
            self.pred.append(temp_pred)
            self.truth.append(temp_truth)

    def accum_noCRF(self, preT, truthT,MASKT):
        #similar with accum(). BUT the input is different due to the output is different between LSTM and CRF
        batch_size, max_seq_len = MASKT.shape
        for i in range(0,batch_size):
            temp_seq_len = MASKT[i].sum().item()
            temp_pred = []
            temp_truth = []
            for j in range(0,temp_seq_len):
                temp_pred.append(labelsmap_inverse[torch.argmax(preT[i][j]).item()])
                temp_truth.append(labelsmap_inverse[truthT[i][j].item()])
            self.pred.append(temp_pred)
            self.truth.append(temp_truth)

    def get_result(self):
        return classification_report(self.truth, self.pred, mode='strict', scheme=IOB2)

def train_loop(dataloader, model, loss_fn, optimizer, nowepoch, device, CRF_en = True):
    size = len(dataloader.dataset)
    model = model.train()
    num_batches = len(dataloader)
    total_loss = 0
    loss=0
    for batch, (X, y, MASK) in enumerate(dataloader):
        if device == 'cuda':
            X = X.cuda()
            y = y.cuda()
            MASK = MASK.cuda()
        pred = model(X)
        if CRF_en:
            loss = model.model_loss(pred,y,MASK)
        else:
            loss = loss_fn(pred.permute(0, 2, 1), y) #the shape of LSTM's output is incompatible with input of nn.CrossEntropyLoss()
        
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(),5) #gradiant clipping
        optimizer.step()

        if batch * len(X) % 1024 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    avg_loss = total_loss / num_batches
    print(f"Train avg loss: {avg_loss:>7f}")


def test_loop(dataloader, model, loss_fn, device, now_epoch, CRF_en = True):
    model = model.eval()
    RS = ResultStat()
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, y, MASK in dataloader:
            if device == 'cuda':
                X = X.cuda()
                y = y.cuda()
                MASK = MASK.cuda()
            pred = model(X)
            if CRF_en:
                test_loss+=model.model_loss(pred,y,MASK)
                bestseq = model.best_seq(pred,MASK)
                RS.accum(bestseq,y,MASK)
            else:
                test_loss +=loss_fn(pred.permute(0,2,1), y).item()
                RS.accum_noCRF(pred,y,MASK)
    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")
    print(RS.get_result())

def main():
    #arguments
    dataset_dir='conll'
    train_batch_size=64 #train set batch size
    test_batch_size=128 #train set batch size
    hidden_dim=200 #hidden state dimension
    emb_dim=100 #embedding dimension
    dpout=0.5 #dropout rate
    learn_rate=5e-2 #learning rate
    epochs=45
    emb_pretrain=True #use pretrained embedding or not
    CRF_enable=True #use pretrained embedding or not
    nll2002_set=False #use conll2002(True) or conll2003(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conll_train = CoNLLset(dataset_dir,nl02=nll2002_set)
    conll_test = CoNLLset(dataset_dir,train=False,nl02=nll2002_set)
    train_loader = DataLoader(conll_train,batch_size=train_batch_size,shuffle=True,collate_fn=conll_collate_fn)
    test_loader = DataLoader(conll_test,batch_size=test_batch_size,shuffle=True,collate_fn=conll_collate_fn)
    model = BiLSTMCRF(hidden_dim=hidden_dim,tag_num=len(labelsmap)-1,embed_dim=emb_dim,emb_pretrain=emb_pretrain,dropout=dpout,nl02=nll2002_set).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=labelsmap['PAD'])
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    optimizer2 = torch.optim.SGD(model.parameters(), lr=learn_rate,weight_decay=1e-4)
    optm=optimizer
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tic = default_timer()
        if(t > 20):
            optm=optimizer2
        train_loop(train_loader, model, loss_fn, optm,t,device,CRF_en = CRF_enable)
        toc = default_timer()
        print('Time: ' + str(toc - tic))
        if t >= 0:
            test_loop(test_loader, model, loss_fn,device,t,CRF_en = CRF_enable)
    print("Done!")

if __name__ == "__main__":
    main()