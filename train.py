import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

def train(dataset, model, max_epochs):
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=dataset.batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(max_epochs):
        state_h, state_c = model.init_state(dataset.sequence_length)

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            #print(x.shape)
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
            if batch%10 == 0:
                print('=====================================================')
                print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

def predict(dataset, model, text, next_words=16):
    words = text
    model.eval()

    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words

if __name__ == "__main__":
    dataset = Dataset(sequence_length=6, batch_size=256, data_path='data/data.smi')
    #model = Model(dataset)
    #train(dataset, model, 100)
    #torch.save(model, './model')
    model = torch.load('./model')
    print(''.join(predict(dataset, model, text=['[','C','H','2',']'])))
