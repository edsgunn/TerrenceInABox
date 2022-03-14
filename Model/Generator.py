import torch.nn as nn
import torch

class LSTM_Generator_Model(nn.Module):
  
    def __init__(self, device, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Generator_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(hidden_size, output_size, 1, batch_first=True, bidirectional=False)
        pass
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        h1 = torch.zeros(1, out.size(0), self.output_size).to(self.device) 
        c1 = torch.zeros(1, out.size(0), self.output_size).to(self.device)
        out, hidden = self.lstm2(out, (h1, c1))

        return out
