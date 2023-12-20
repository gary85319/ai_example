import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import jieba
import time
import re
with open("r", "r", encoding="utf-8") as file:
    text = file.read()
    text = re.sub(r'[\n,，。\t()]+', '', text)
    


vocab = list(set(jieba.cut(text)))
vocab.sort()
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}
or_vacad=list(jieba.cut(text))
text_as_int = np.array([char_to_idx[char] for char in or_vacad])

class CharDataset(Dataset):
    
    def __init__(self, text_as_int, seq_length):
        self.seq_length = seq_length
        self.text_as_int = text_as_int

    def __len__(self):
        return len(self.text_as_int) - self.seq_length

    def __getitem__(self, idx):
        input_seq = torch.LongTensor(self.text_as_int[idx:idx+1])
        target = torch.LongTensor(self.text_as_int[idx+1:idx+self.seq_length+1])
        return input_seq, target

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super(CharLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size*seq_length)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x[:, -1, :])
        return x, hidden
seq_length = 25
batch_size = 128
vocab_size = len(vocab)
embedding_dim = 128
hidden_size = 256
output_size = len(vocab)
num_layers=4
load_model=False
char_dataset = CharDataset(text_as_int, seq_length)
data_loader = DataLoader(char_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
model = CharLSTM(vocab_size, embedding_dim, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
last_time=time.time()
num_epochs = 1800
plot_every = 300
current_loss = 0
all_losses = []
c=[]
a=0
if not load_model:
    for n in range(num_epochs):
        
    
        for i, (inputs, labels) in enumerate(data_loader):
            hidden = None
            optimizer.zero_grad()
            
            outputs, hidden = model(inputs, hidden)

            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            all_losses.append(loss.item())
            print(f'iter:[{i}/{len(data_loader)}]loss:{loss.item()},epoch:[{n}/{num_epochs}]',end='\r')
            a=a+1
            c.append(a)
            optimizer.step()
else:
    with open('chat_mode2.dat', 'rb') as f:
        
        model = torch.load(f, map_location=torch.device('cpu'))
def generate_text(model, start_text, length=100, len_seq=20):
    if load_model:
        if isinstance(model, CharLSTM):
            model.eval()
        else:
            print("模型不合")
            sys.exit()
    
    model.eval()
    generated_text = start_text

    
    input_seq = torch.LongTensor([char_to_idx[char] for char in jieba.cut(start_text)])
    input_seq = input_seq.unsqueeze(0)
    print(input_seq)
    output, _ = model(input_seq, None)
    output = output.view(-1, len(vocab))
    probabilities = F.softmax(output, dim=1)
    predicted_idx = torch.argmax(output, dim=1).detach().numpy()
    
    for n in predicted_idx:
        predicted_char = idx_to_char[n]
        generated_text +=  predicted_char
    print(f'訓練時長:{time.time()-last_time}')
    print(generated_text)
generate_text(model, start_text="我", len_seq=20, length=100)
if not load_model:
    torch.save(model, 'chat_mode2.dat')
    plt.plot(c,all_losses)
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.title('history Training Loss over Iterations')
    plt.show()
while True:
    generate_text(model, start_text=input('輸入文字:'), len_seq=30, length=100)




