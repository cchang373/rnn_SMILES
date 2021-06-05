import torch
from torch import LongTensor
import torch.nn as nn
from torch.nn import Embedding, LSTM, Linear
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Dataset():
    def __init__(self, data_path):
        self.data_path = data_path
        self.seqs = self.get_seqs()
        self.vocab = self.get_vocab()
        self.vectorized_seqs = self.get_vectorized_seqs()
        self.seq_lengths = self.get_seq_lengths()
        self.seq_tensor = self.get_seq_tensor()
        self.seq_lengths_sorted, self.seq_tensor_sorted = self.get_sorted()

    def get_seqs(self):
        train_df = open(self.data_path).read()
        seqs = train_df.split('\n')[:5000]
        return seqs
    
    def get_vocab(self):
        vocab =  ['<pad>']+sorted(set([char for seq in self.seqs for char in seq]))
        return vocab

    def get_vectorized_seqs(self):
         vectorized_seqs = [[self.vocab.index(tok) for tok in seq] for seq in self.seqs]
         return vectorized_seqs

    def get_seq_lengths(self):
        seq_lengths = LongTensor(list(map(len, self.vectorized_seqs)))
        return seq_lengths

    def get_seq_tensor(self):
        seq_tensor = Variable(torch.zeros((len(self.vectorized_seqs), self.seq_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(self.vectorized_seqs, self.seq_lengths)):
            seq_tensor[idx, :seqlen] = LongTensor(seq)
        return seq_tensor
    
    def get_sorted(self):
        seq_lengths_sorted, perm_idx = self.seq_lengths.sort(0, descending=True)
        seq_tensor_sorted = self.seq_tensor[perm_idx]
        return seq_lengths_sorted, seq_tensor_sorted

class Model(nn.Module):
    def __init__(self, data, embedding_dim, input_size, hidden_size):
        super(Model, self).__init__()
        self.data = data
        self.embed = Embedding(len(self.data.vocab), embedding_dim)
        self.lstm = LSTM(input_size = input_size,
                hidden_size = hidden_size,
                batch_first = True)
        self.packed_input = self.pack_input()
        self.linear = Linear(hidden_size, len(self.data.vocab))

    def pack_input(self):
        embedded_seq_tensor = self.embed(self.data.seq_tensor_sorted)
        packed_input = pack_padded_sequence(embedded_seq_tensor, self.data.seq_lengths_sorted.cpu().numpy(), batch_first = True)
        return packed_input

    def forward(self, packed_input):
        packed_output, (ht, ct) = self.lstm(packed_input)
        output = self.linear(packed_output.data)
        return output, (ht, ct)

if __name__ == "__main__":
    seq_tensor = Dataset('data/data.smi')
    model = Model(data=seq_tensor, 
        embedding_dim=3, 
        input_size=3, 
        hidden_size=5)
    output, (ht, ct) = model.forward(model.packed_input)
    print(output.data.shape)


