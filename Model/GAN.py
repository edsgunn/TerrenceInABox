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

discriminator = DiscriminatorModel()
generator = GeneratorModel()
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
        noise = torch.randn(batch_size,100).to(device)
        # Forward pass         
        generated_data = generator(noise) # batch_size X 784
        
        true_data = data_input[0].view(batch_size, 784).to(device) # batch_size X 784
        digit_labels = data_input[1] # batch_size
        true_labels = torch.ones(batch_size).to(device)
        
        # Clear optimizer gradients        
        discriminator_optimizer.zero_grad()
        # Forward pass with true data as input
        discriminator_output_for_true_data = discriminator(true_data).view(batch_size)
        # Compute Loss
        true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels)
        # Forward pass with generated data as input
        discriminator_output_for_generated_data = discriminator(generated_data.detach()).view(batch_size)
        # Compute Loss 
        generator_discriminator_loss = loss(
            discriminator_output_for_generated_data, torch.zeros(batch_size).to(device)
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
        generated_data = generator(noise) # batch_size X 784
        # Forward pass with the generated data
        discriminator_output_on_generated_data = discriminator(generated_data).view(batch_size)
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
                noise = torch.randn(batch_size,100).to(device)
                generated_data = generator(noise).cpu().view(batch_size, 28, 28)
                for x in generated_data:
                    plt.imshow(x.detach().numpy(), interpolation='nearest',cmap='gray')
                    plt.show()

                    break


    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch_idx), training_parameters["n_epochs"], torch.mean(torch.FloatTensor(D_loss)), torch.mean(torch.FloatTensor(G_loss))))