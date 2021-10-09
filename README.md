# rnn_SMILES
lstm for SMILES notation generation

## Usage
```
from model import Model
from dataset import Dataset
from train import train, predict

dataset = Dataset(sequence_length=6, batch_size=64, data_path='data/data.smi') #load the dataset
model = Model(dataset, 64, 128, 3, 0.2) #Model takes the input (dataset, lstm_size, embedding_size, num_layers, dropout ratio)
train(dataset, model, 100) #train the model with (dataset, model, max_epochs)
torch.save(model, './model') #save the model
#model = torch.load('./model') #reload the trained model
print(''.join(predict(dataset, model, text=['[','C','H','2',']']))) #generate new SMILES notations from the starting point
```
