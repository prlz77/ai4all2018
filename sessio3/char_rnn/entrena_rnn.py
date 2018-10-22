#!/usr/bin/env python
# coding: utf-8
import argparse

import numpy as np
import torch
import torch.nn as nn
import unidecode
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def preprocess(text):
    """Receives a text and reduces the amount of synbols. Passes to lowercase.
    Args:
        text: input text to transform
    """
    text = unidecode.unidecode(text)
    text = text.replace('\n', ' ').replace('\ufeff', '').lower()
    text = ''.join(x for x in text if x not in "%$#=<>/*+@][")
    while len(text) != len(text.replace("  ", " ")):
        text = text.replace("  ", " ")
    return text


class BaseDades(Dataset):
    def __init__(self, ruta="./data/quijote.txt", mida_sequencia=5, percent_inici=0, percent_fi=0.8):
        """ Constructor
        Args:
            ruta: path to load input text file
            mida_sequencia: sequence length
            percent_inici: start of text subset
            percent_fi: end of text subset
        """
        with open("data/quijote.txt", 'r') as infile:
            data = preprocess(infile.read())
        self.alphabet = sorted(set(data))
        mida = len(data)
        data = data[int(mida * percent_inici):int(mida * percent_fi)]
        data_int = [self.alphabet.index(x) for x in data]
        self.finestres = []

        for i in range(len(data_int) - mida_sequencia):
            seq = torch.LongTensor(data_int[i:(i + mida_sequencia)])
            self.finestres.append(seq)
            if i % 100000 == 0:
                print("%d/%d" % (i, len(data_int)))

    def __getitem__(self, idx):
        """ Pytorch asks to implement this method, returns the batch with index idx.
        Args:
            idx: index of the minibatch to return
        """
        if idx == len(self.finestres) - 1:
            idx = 0
        return self.finestres[idx], self.finestres[idx + 1]

    def __len__(self):
        return len(self.finestres)


class Recurrent(nn.Module):
    def __init__(self, mida_alfabet, n_neurones, n_capes):
        """ Recurrent neural network definition
        Args:
            mida_alfabet: alphabet size
            n_neurones: n neurons
            n_capes: number of recurrent layers
        """
        super().__init__()
        self.n_neurones = n_neurones
        self.n_capes = n_capes
        self.mida_alfabet = mida_alfabet
        self.rnn = nn.GRU(n_neurones, n_neurones, n_capes, batch_first=True, dropout=0.3)
        self.linear_output = nn.Linear(n_neurones, mida_alfabet, bias=False)

    def forward(self, x):
        """ Network forward function
        Args:
            x: network input (batch, sequence length, channels)
        """
        b, s = x.size()
        y = nn.functional.embedding(x, self.linear_output.weight)
        estat = torch.zeros(self.n_capes, x.size(0), self.n_neurones).cuda()
        y, estat = self.rnn(y, estat)
        y = y.contiguous().view(b * s, self.n_neurones)
        return self.linear_output(y).view(b, s, self.mida_alfabet)


def main(args):
    """
    Args:
        args:
    """
    mida_batch = 256
    mida_seq = args.mida_seq
    base_dades_train = BaseDades(args.fitxer_txt, mida_sequencia=mida_seq)
    base_dades_test = BaseDades(args.fitxer_txt, mida_sequencia=mida_seq, percent_inici=0.8, percent_fi=1.0)
    lector_dades_train = DataLoader(base_dades_train, mida_batch, shuffle=True, num_workers=4, pin_memory=True)
    lector_dades_test = DataLoader(base_dades_test, mida_batch, shuffle=False, num_workers=4, pin_memory=True)

    mida_alfabet = len(base_dades_train.alphabet)
    n_neurones = 256
    n_capes = 2
    net = Recurrent(mida_alfabet, n_neurones, n_capes).cuda()
    entrada_prova = torch.LongTensor([[2, 3, 4], [1, 3, 2]]).cuda()
    net(entrada_prova)

    coef_aprenentatge = 0.001
    optimitzador = torch.optim.Adam(net.parameters(), coef_aprenentatge)

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

    best_loss = np.inf
    no_improvement_count = 0
    for it in range(args.max_iter):
        print("Iter", it, "de", args.max_iter)
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
    parser.add_argument("--max_iter", type=int, default=100)
    args = parser.parse_args()
    main(args)
