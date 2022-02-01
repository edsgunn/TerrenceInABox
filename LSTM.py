#https://medium.com/@nutanbhogendrasharma/pytorch-recurrent-neural-networks-with-mnist-dataset-2195033b540f
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch import optim as optim
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_parameters = {
    "n_epochs": 100,
    "batch_size": 100,
}
data_loader = torch.utils.data.DataLoader(
  
  datasets.MNIST('./', train=True, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.5,), (0.5,))
                             ])),
  batch_size=training_parameters["batch_size"], shuffle=True)


class LSTMModel(nn.Module):
  
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        pass
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out
       
        pass
pass
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
learning_rate = 0.01
loss = nn.CrossEntropyLoss()
generator = LSTMModel(input_size, hidden_size, num_layers, num_classes)
generator.to(device)
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
batch_size = training_parameters["batch_size"]

for epoch_idx in range(training_parameters["n_epochs"]):
    LSTM_loss = []
    for batch_idx, data_input in enumerate(data_loader):

        data = data_input[0].reshape(-1, sequence_length, input_size).to(device)
        digit_labels = data_input[1] # batch_size

        generated_data = generator(data)#.view(batch_size)
        generator_loss = loss(generated_data, digit_labels)

        generator_optimizer.zero_grad()

        generator_loss.backward()
        generator_optimizer.step()

        LSTM_loss.append(generator_loss.data.item())

    print('[%d/%d]: loss_LSTM: %.3f' % (
            (epoch_idx), training_parameters["n_epochs"], torch.mean(torch.FloatTensor(LSTM_loss))))