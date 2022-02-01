##Made following the tutorial at https://medium.com/analytics-vidhya/step-by-step-implementation-of-conditional-generative-adversarial-networks-54e4b47497d6

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


class GeneratorModel(nn.Module):
  
  def __init__(self):
    super(GeneratorModel, self).__init__()
    input_dim = 100
    output_dim = 784
    self.hidden_layer1 = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.LeakyReLU(0.2)
    )
    self.hidden_layer2 = nn.Sequential(
    nn.Linear(256, 512),
    nn.LeakyReLU(0.2)
    )
    self.hidden_layer3 = nn.Sequential(
    nn.Linear(512, 1024),
    nn.LeakyReLU(0.2)
    )
    self.output_layer = nn.Sequential(
    nn.Linear(1024, output_dim),
    nn.Tanh()
    )
    
  def forward(self, x):
    output = self.hidden_layer1(x)
    output = self.hidden_layer2(output)
    output = self.hidden_layer3(output)
    output = self.output_layer(output)
    return output.to(device)


class DiscriminatorModel(nn.Module):
  
  def __init__(self):
    super(DiscriminatorModel, self).__init__()
    input_dim = 784
    output_dim = 1
    self.hidden_layer1 = nn.Sequential(
    nn.Linear(input_dim, 1024),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3)
    )
    self.hidden_layer2 = nn.Sequential(
    nn.Linear(1024, 512),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3)
    )
    self.hidden_layer3 = nn.Sequential(
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3)
    )
    self.output_layer = nn.Sequential(
    nn.Linear(256, output_dim),
    nn.Sigmoid()
    )
    
  def forward(self, x):
    output = self.hidden_layer1(x)
    output = self.hidden_layer2(output)
    output = self.hidden_layer3(output)
    output = self.output_layer(output)
    return output.to(device)

class LSTM_Discriminator_Model(nn.Module):
  
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Discriminator_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
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

class LSTM_Generator_Model(nn.Module):
  
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Generator_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
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


sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
output_size = 28*28
learning_rate = 0.01
discriminator = LSTM_Discriminator_Model(input_size, hidden_size, num_layers, 1)
generator = LSTM_Generator_Model(input_size, hidden_size, num_layers, output_size)
discriminator.to(device)
generator.to(device)
loss = nn.BCELoss()
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002)



for epoch_idx in range(training_parameters["n_epochs"]):
    G_loss = []
    D_loss = []
    for batch_idx, data_input in enumerate(data_loader):
        
        # Generate noise and move it the device
        noise = torch.randn(training_parameters["batch_size"],sequence_length,input_size).to(device)
        # Forward pass         
        generated_data = generator(noise).view(training_parameters["batch_size"],sequence_length,input_size) # batch_size X 784
        
        true_data = data_input[0].view(training_parameters["batch_size"], sequence_length, input_size).to(device) # batch_size X 784
        digit_labels = data_input[1] # batch_size
        true_labels = torch.ones(training_parameters["batch_size"]).to(device)
        
        # Clear optimizer gradients        
        discriminator_optimizer.zero_grad()
        # Forward pass with true data as input
        discriminator_output_for_true_data = discriminator(true_data).view(training_parameters["batch_size"])
        # Compute Loss
        true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels)
        # Forward pass with generated data as input
        discriminator_output_for_generated_data = discriminator(generated_data.detach()).view(training_parameters["batch_size"])
        # Compute Loss 
        generator_discriminator_loss = loss(
            discriminator_output_for_generated_data, torch.zeros(training_parameters["batch_size"]).to(device)
        )
        # Average the loss
        discriminator_loss = (
            true_discriminator_loss + generator_discriminator_loss
        ) / 2
               
        # Backpropagate the losses for Discriminator model      
        discriminator_loss.backward()
        discriminator_optimizer.step()

        D_loss.append(discriminator_loss.data.item())
        
        
        # Clear optimizer gradients

        generator_optimizer.zero_grad()
        
        # It's a choice to generate the data again
        generated_data = generator(noise).view(training_parameters["batch_size"],sequence_length,input_size) # batch_size X 784
        # Forward pass with the generated data
        #print(generated_data.size())
        discriminator_output_on_generated_data = discriminator(generated_data).view(training_parameters["batch_size"])
        # Compute loss
        generator_loss = loss(discriminator_output_on_generated_data, true_labels)
        # Backpropagate losses for Generator model.
        generator_loss.backward()
        generator_optimizer.step()
        
        G_loss.append(generator_loss.data.item())
        # Evaluate the model
        if ((batch_idx + 1)% 500 == 0 and (epoch_idx + 1)%10 == 0):
            print("Training Steps Completed: ", batch_idx)
            
            with torch.no_grad():
                noise = torch.randn(training_parameters["batch_size"],sequence_length,input_size).to(device)
                generated_data = generator(noise).cpu().view(training_parameters["batch_size"], 28, 28)
                for x in generated_data:
                    plt.imshow(x.detach().numpy(), interpolation='nearest',cmap='gray')
                    plt.show()

                    break


    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch_idx), training_parameters["n_epochs"], torch.mean(torch.FloatTensor(D_loss)), torch.mean(torch.FloatTensor(G_loss))))