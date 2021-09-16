# rnn_SMILES
lstm for SMILES notation generation

## Usage
```
from model import Model
from dataset import Dataset
from train import train, predict

dataset = Dataset(sequence_length=6, batch_size=256, data_path='data/data.smi')
model = Model(dataset, 128, 64, 3, 0.2)
train(dataset, model, 100)
torch.save(model, './model')
#model = torch.load('./model')
print(''.join(predict(dataset, model, text=['[','C','H','2',']'])))
```
