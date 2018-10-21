#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import unidecode
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse


def preprocess(text):
    text = unidecode.unidecode(text)
    text = text.replace('\n', ' ').replace('\ufeff', '').lower()
    text = ''.join(x for x in text if x not in "%$#=<>/*+@][")
    while len(text) != len(text.replace("  ", " ")):
        text = text.replace("  ", " ")
    return text


def main(args):
    class BaseDades(Dataset):
        def __init__(self, ruta="./data/quijote.txt", mida_sequencia=5, percent_inici=0, percent_fi=0.8):
            with open("data/quijote.txt", 'r') as infile:
                data = preprocess(infile.read())
            self.alphabet = sorted(set(data))
            mida = len(data)
            data = data[int(mida*percent_inici):int(mida*percent_fi)]
            data_int = [self.alphabet.index(x) for x in data]
            self.finestres = []

            for i in range(len(data_int) - mida_sequencia):
                seq = torch.LongTensor(data_int[i:(i + mida_sequencia)])
                self.finestres.append(seq)
                if i % 100000 == 0:
                    print("%d/%d" %(i, len(data_int)))  
                    
        def __getitem__(self, idx):
            if idx == len(self.finestres) - 1:
                idx = 0
            return self.finestres[idx], self.finestres[idx + 1]
        
        def __len__(self):
            return len(self.finestres)
            
    mida_batch = 256
    mida_seq = 16
    base_dades_train = BaseDades(mida_sequencia=mida_seq)
    base_dades_test = BaseDades(mida_sequencia=mida_seq, percent_inici=0.8, percent_fi=1.0)
    lector_dades_train = DataLoader(base_dades_train, mida_batch, shuffle=True, num_workers=1, pin_memory=True)
    lector_dades_test = DataLoader(base_dades_test, mida_batch, shuffle=False, num_workers=1, pin_memory=True)
        
        


    # In[2]:


    class Recurrent(nn.Module):
        def __init__(self, mida_alfabet, n_neurones, n_capes):
            super().__init__()
            self.n_neurones = n_neurones
            self.n_capes = n_capes
            self.mida_alfabet = mida_alfabet
            self.rnn = nn.GRU(n_neurones, n_neurones, n_capes, batch_first=True, dropout=0.3)
            self.linear_output = nn.Linear(n_neurones, mida_alfabet, bias=False)
        
        def forward(self, x):
            b, s = x.size()
            y = nn.functional.embedding(x, self.linear_output.weight)
            estat = torch.zeros(self.n_capes, x.size(0), self.n_neurones).cuda()
            y, estat = self.rnn(y, estat)
            y = y.contiguous().view(b * s, self.n_neurones)
            return self.linear_output(y).view(b, s, self.mida_alfabet)

    mida_alfabet = len(base_dades_train.alphabet)
    n_neurones = 256
    n_capes = 2
    net = Recurrent(mida_alfabet, n_neurones, n_capes).cuda()
    entrada_prova = torch.LongTensor([[2,3,4],[1,3,2]]).cuda()
    net(entrada_prova)

    coef_aprenentatge = 0.001
    optimitzador = torch.optim.Adam(net.parameters(), coef_aprenentatge)


    # In[3]:


    def entrena():
        net.train()
        train_loss = 0
        for entrada, sortida in tqdm(lector_dades_train):
            entrada = entrada.cuda()
            sortida = sortida.cuda()
            optimitzador.zero_grad()
            b, s = sortida.size()
            sortida_xarxa = net(entrada).view(b * s, -1)
            
            sortida = sortida.view(b * s)
            loss = nn.functional.cross_entropy(sortida_xarxa, sortida)
            loss.backward()
            optimitzador.step()
            train_loss = train_loss * 0.1 + float(loss) * 0.9
        return train_loss
            
    def testeja():
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for entrada, sortida in tqdm(lector_dades_test):
                entrada = entrada.cuda()
                sortida = sortida.cuda()
                b, s = sortida.size()
                sortida_xarxa = net(entrada).view(b * s, -1)
                sortida = sortida.view(b * s)
                loss = nn.functional.cross_entropy(sortida_xarxa, sortida)
                val_loss += loss
        return val_loss / len(lector_dades_test)

    iteracions = 100
    best_loss = np.inf
    no_improvement_count = 0
    for it in range(iteracions):    
        print("Iter", it, "de", iteracions)
        print("Loss train", entrena())
        val_loss = testeja()
        print("Loss val", val_loss)
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save({'net': net.state_dict(), 'alphabet': base_dades_train.alphabet}, "char_rnn.pth")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count == 10:
                break




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fitxer_txt", type=str, default="quijote.txt")
    parser.add_argument("--mida_seq", type=int, default=16)
    args = parser.parse_args()
    main(args)

