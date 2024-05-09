### RNN �� ���� ###

### BasicGRU��� RNN �� ���� ###

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGRU(nn.Module):
    ### Q. �� ������ ��������� ���Ǵµ� � �ǹ̸� ��������? (nn.Module, super~~)
    def __init__(self,n_layers,hidden_dim,n_vocab,embed_dim,n_classes,dropout_p=0.2):
        super(BasicGRU,self).__init__()
        print("Building Basic GRU model...")

        self.n_layers = n_layers # ���� ���͵��� '��' ���� ����
        self.embed = nn.Embedding(n_vocab,embed_dim) #n_vocab�� vocab�� ������, embed_dim�� �Ӻ��� �� �ܾ� �ټ��� ���ϴ� ������
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim,self.hidden_dim,num_layers=self.n_layers,batch_first=True) #GRU�� LSTM�� ���������� RNN������ ���� �ҽ� or ���� ���� ������ �ذ��ϴ� ��Ŀ������ ������ �ִ�.

        # gru�� ��ģ ������� �ϳ��� ����� ���ͷ� ��ȯ�ȴ�.
        # �ش� ����(����)�� ���� ��ȭ ���䰡 ���������� ���������� �з��ϱ� ���� ���຤�͸� �Ʒ��� ���� �Ű���� ����ϰ� �Ѵ�.
        self.out = nn.Linear(self.hidden_dim,n_classes)

    # h_0�� ����� ���� �Լ�, �ʱ⿡�� 0���� �ʱ�ȭ �ȴ�.
    def _init_state(self,batch_size=1):
        weight = next(self.parameters()).data # self.parameters()�� �Ű�� ���(nn.Module)�� ����ġ �������� �ݺ��� ���·� ��ȯ
                                              # �ش� �ݺ��ڰ� �����ϴ� ���ҵ��� ���� �Ű���� ����ġ �ټ�(.data)�� ���� ��ü��
                                              # �� �� ��ɾ�� nn.GRU ���� ù ��° ����ġ �ټ��� ����
        return weight.new(self.n_layers,batch_size,self.hidden_dim).zero_() # ����� ����ġ �ټ��� (n_layers,batch_size,hidden_dim) ����� ���� �ټ��� ��ȯ �� �ټ��� ��� ���� 0���� �ʱ�ȭ
        ### Q. ���� �ټ��� ����� �ٲٴ� ����?? (���� input�̳� embedding�� input�� ����ȭ �Ǿ� �ִ� ���ΰ�?)

    def forward(self,x):
        x = self.embed(x) # �Էµ����� x(�� ��ġ �ӿ� �ִ� ��� ��ȭ��)�� ������ �迭�� ��ȯ
        h_0 = self._init_state(batch_size = x.size(0)) # �ٸ� �Ű����� �޸� RNN�� �Է� ������ �ܿ��� ù ��° ���� ���� H0�� ������ x�� �Բ� �־���� �Ѵ�.
        x, _ = self.gru(x,h_0) # self.gru�� ������� (batch_size,x�� length,hidden_dim) ����� ������ 3d �ټ�
        h_t = x[:,-1,:] # ������ ���к��͸� �����Ͽ� h_t�� ����
        self.dropout(h_t) ### Q.dropout()�� �� h_t�� ���� ���ΰ�? dropout�� �Ϻ� ������ drop��Ű�°� �ƴϿ���?

        logit = self.out(h_t)
        return logit
    
    ### �н��Լ��� ���Լ�

    def train(model,optimizer,train_iter):
        model.train() # ���� �н����� ��ȯ
        for b,batch in enumerate(train_iter): 
            x,y = batch.text.to(DEVICE), batch.label.to(DEVICE) # batch�� ���� train_iter�� text�� label �����͸� x��y�� ����
            y.data.sub_(1) # ���Ǹ� ���� �󺧸� [1,2] -> [0,1]
            optimizer.zero_grad() # ���� ������ ���ϱ� ���� �ʱ�ȭ
            logit = model(x) # ���� ����� logit��, ���� ����� �Ǽ� ���̰� ��ġ��ŭ�� ������ ����
            loss = F.cross_entropy(logit,y) # logit�� ���� ��(y)�� cross entropy�� ���� �սǰ� ���
            loss.backward() # ������
            optimizer.step() # ����ġ ����

    def evaluate(model,val_iter):
        """evaluate model"""
        model.eval()
        corrects, total_loss = 0,0
        for batch in val_iter:
            x,y = batch.text.to(DEVICE), batch.label.to(DEVICE)
            y.data.sub_(1)
            logit = model(x)
            loss = F.cross_entropy(logit,y,reduction='sum')
            total_loss += loss.item()
            corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
        
        size = len(val_iter.dataset)
        avg_loss = total_loss / size
        avg_accuray = 100.0*corrects/size
        return avg_loss,avg_accuray
    
    ### �� ��ü ����
    # �� ���к��� ������ = 256 / �Ӻ����� ��ū�� ������ 128
    model = BasicGRU(1,256,vocab_size,128,n_classes,0.5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr) # ����ȭ �˰��� ����


    ### �н� ����
    best_val_loss = None
    for e in range(1,EPOCHS+1):
        train(model,optimizer,train_iter)
        val_loss, val_accuracy = evaluate(model,val_iter)

        print("[���� : %d] ���� ���� : %5.2f | ���� ��Ȯ�� : %5.2f" %(e,val_loss,val_accuracy))

    # ���� ���� �н� ������ �ƴ� '���� ����'�� �ּ�ȭ�� ���� ���õȴ�.
    # ���� ������ ���� ���� ���� ����
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

    