import torch.nn as nn
import torch

class LSTM_Generator_Model(nn.Module):
  
    def __init__(self, device, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Generator_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device
        self.fc_layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc_layer_2 = nn.Linear(2*hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        pass
    

    def forward(self, x):
        # Set initial hidden and cell states 
        # h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        # c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # Passing in the input and hidden state into the model and  obtaining outputs
        out = self.fc_layer_1(x)
        out = self.dropout1(out)
        out = self.relu(out)
        out, hidden = self.lstm(out)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.tanh(out)
        out = self.fc_layer_2(out)
        out = self.softmax(out)

        return out
