import torch.nn as nn
import torch

class LSTM_Discriminator_Model(nn.Module):
  
    def __init__(self, device, input_size, hidden_size, seq_len, num_layers, output_size):
        super(LSTM_Discriminator_Model, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = 0.2, bidirectional=True)
        self.tanh = nn.Tanh()
        # self.fc_layer = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(2*hidden_size*seq_len, output_size),
            nn.Sigmoid()
        )
        pass
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.tanh(out)
        # out = self.fc_layer(out)
        # out = self.sigmoid(out)
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = torch.reshape(out,(-1,2*self.hidden_size*self.seq_len))
        out = self.fc(out)
        return out
