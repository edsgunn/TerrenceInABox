##Made following the tutorial at https://medium.com/analytics-vidhya/step-by-step-implementation-of-conditional-generative-adversarial-networks-54e4b47497d6

import os
import argparse 
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import optim as optim
from matplotlib import pyplot as plt
from DataLoader import MusicDataset
from Discriminator import LSTM_Discriminator_Model
from Generator import LSTM_Generator_Model
from torchviz import make_dot
import sys
sys.path.append(f'{os.getcwd()}/Interface')
from GeneratorPlayer import accompanySong

def train_model(model, opt):


    print("Running on:",opt.device)

    print("Loading Dataset")
    data_loader = torch.utils.data.DataLoader(opt.dataset,batch_size=opt.batch_size, shuffle=True)

    sequence_length = opt.dataset.max_length
    discriminator = model["Discriminator"]
    generator = model["Generator"]
    discriminator.to(opt.device)
    generator.to(opt.device)
    loss = nn.BCELoss()#reduction='mean')
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0005)

    print("Starting Training")
    Overall_G_Loss = []
    Overall_D_loss = []
    for epoch_idx in range(opt.epochs):
        G_loss = []
        D_loss = []
        for batch_idx, data_input in enumerate(data_loader):
            
            # Generate noise and move it the device
            classes = data_input["Melody"].to(opt.device)
            batch_size = classes.size(dim=0)
            noise = torch.randn(batch_size,sequence_length,opt.noise_size).to(opt.device)
            noise = torch.cat((noise,classes),2)
            # Forward pass 
            generated_data = generator(noise)
            generated_data = generated_data.view(batch_size,sequence_length,opt.output_size)
            if opt.one_hot:
                generated_data = generated_data.argmax(2)
                generated_data = f.one_hot(generated_data, num_classes = opt.output_size)
            generated_data = torch.cat((generated_data,classes),2)

            true_data = data_input["Melody"].view(batch_size, sequence_length, opt.input_size)
            digit_labels = 0.9*data_input["Chords"].view(batch_size,sequence_length,opt.output_size)
            true_data = torch.cat((true_data,digit_labels),2).to(opt.device)
            true_labels = torch.ones(batch_size).to(opt.device)#, sequence_length).to(opt.device)
            
            # Clear optimizer gradients        
            discriminator_optimizer.zero_grad()
            # Forward pass with true data as input
            discriminator_output_for_true_data = discriminator(true_data).view(batch_size)#, sequence_length)
            # Compute Loss
            true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels)
            # Forward pass with generated data as input
            discriminator_output_for_generated_data = discriminator(generated_data.detach()).view(batch_size)#, sequence_length)
            # Compute Loss 
            generator_discriminator_loss = loss(
                discriminator_output_for_generated_data, torch.zeros(batch_size).to(opt.device)#, sequence_length).to(opt.device)
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
            generated_data = generator(noise).view(batch_size, sequence_length, opt.output_size)
            if opt.one_hot:
                generated_data = generated_data.argmax(2)
                generated_data = f.one_hot(generated_data, num_classes = opt.output_size)   
            generated_data = torch.cat((generated_data,classes),2)
            # Forward pass with the generated data
            discriminator_output_on_generated_data = discriminator(generated_data).view(batch_size)#, sequence_length)
            # Compute loss
            generator_loss = loss(discriminator_output_on_generated_data, true_labels)
            # Backpropagate losses for Generator model.
            generator_loss.backward()
            generator_optimizer.step()
            
            G_loss.append(generator_loss.data.item())
            # Evaluate the model
            if ((batch_idx  == 0) and (epoch_idx + 1)%opt.printevery == 0):                
                with torch.no_grad():
                    classes = data_input["Melody"]
                    batch_size = classes.size(dim=0)
                    noise = torch.randn(batch_size,sequence_length,opt.noise_size)
                    noise = torch.cat((noise,classes),2).to(opt.device)
                    generated_data = generator(noise).cpu().view(batch_size, sequence_length, opt.output_size)
                    # make_dot(noise, params=dict(list(generator.named_parameters()))).render("Generator_torchviz", format="png")
                    # y = discriminator(torch.cat((data_input["Melody"],data_input["Chords"]),2))
                    # make_dot(y, params=dict(list(discriminator.named_parameters()))).render("Discriminator_torchviz", format="png")
                    # accompanySong(generator,"DON'T STOP BELIEVIN'")
                    for x in generated_data:
                        print(x[1,:])
                        break

        mean_D_loss = torch.mean(torch.FloatTensor(D_loss))
        mean_G_loss = torch.mean(torch.FloatTensor(G_loss))
        Overall_G_Loss.append(mean_G_loss)
        Overall_D_loss.append(mean_D_loss)
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                (epoch_idx), opt.epochs, mean_D_loss, mean_G_loss))
    x = [i for i in range(opt.epochs)]
    if opt.save:
        max_path_index = 0
        for file in os.listdir(opt.save_dir+"/generators"):
            max_path_index = max(max_path_index,int(file.split('_')[-1][:-3]))
        torch.save(generator, f"{opt.save_dir}/generators/generator_{max_path_index+1}.pt")
        max_path_index = 0
        for file in os.listdir(opt.save_dir+"/discriminators"):
            max_path_index = max(max_path_index,int(file.split('_')[-1][:-3]))
        torch.save(discriminator, f"{opt.save_dir}/discriminators/discriminator_{max_path_index+1}.pt")
    plt.plot(x, Overall_D_loss, label = "Discriminator Loss")
    plt.plot(x, Overall_G_Loss, label = "Generator Loss")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.show()
    classes = data_input["Melody"]
    batch_size = classes.size(dim=0)
    noise = torch.randn(batch_size,sequence_length,opt.noise_size)
    noise = torch.cat((noise,classes),2).to(opt.device)
    generated_data = generator(noise).cpu().view(batch_size, sequence_length, opt.output_size)
    # make_dot(noise, params=dict(list(generator.named_parameters()))).render("Generator_torchviz", format="png")
    # y = discriminator(torch.cat((data_input["Melody"],data_input["Chords"]),2))
    # make_dot(y, params=dict(list(discriminator.named_parameters()))).render("Discriminator_torchviz", format="png")
    # accompanySong(generator,"DON'T STOP BELIEVIN'")
    i = 0
    for x in generated_data:
        print(x)
        print(x.size())
        i +=1
        if i == 3: break

# training_parameters = {
#     "n_epochs": 10,
#     "batch_size": 10,
# }

# hidden_size = 256
# num_layers = 2
# noise_size = 24
# max_sequence_length = 1000

def train_LSTM(opt):
    print("Running on:",opt.device)

    print("Loading Dataset")
    data_loader = torch.utils.data.DataLoader(opt.dataset,batch_size=opt.batch_size, shuffle=True)

    sequence_length = opt.dataset.max_length
    generator = LSTM_Generator_Model(opt.device ,opt.input_size, opt.h_size, opt.n_layers, opt.output_size)
    generator.to(opt.device)
    loss = nn.MSELoss(reduction='mean')
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    

    print("Starting Training")
    Overall_G_Loss = []
    for epoch_idx in range(opt.epochs):
        G_loss = []
        for batch_idx, data_input in enumerate(data_loader):
            classes = data_input["Melody"].to(opt.device)
            batch_size = classes.size(dim=0)
            # Forward pass     
            generated_data = generator(classes)
            generated_data = generated_data.view(batch_size,sequence_length,opt.output_size)
            chords = data_input["Chords"].view(batch_size,sequence_length,opt.output_size)
            # print(generated_data.size(), chords.size())
            generator_loss = loss(generated_data, chords)

            generator_loss.backward()
            generator_optimizer.step()

            G_loss.append(generator_loss.data.item())
            # Evaluate the model
            if ((batch_idx  == 0) and (epoch_idx + 1)%opt.printevery == 0):                
                with torch.no_grad():
                    classes = data_input["Melody"]
                    batch_size = classes.size(dim=0)
                    generated_data = generator(classes).cpu().view(batch_size, sequence_length, opt.output_size)
                    for x in generated_data:
                        print(x[1,:])
                        break

        mean_G_loss = torch.mean(torch.FloatTensor(G_loss))
        Overall_G_Loss.append(mean_G_loss)
        print('[%d/%d]: loss_g: %.3f' % (
                (epoch_idx), opt.epochs, mean_G_loss))

        
    plt.plot(x, Overall_G_Loss, label = "Generator Loss")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.show()



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-input_size', type=int, default=12)                            #Size of melody vector (shouldn't be specified often)
    parser.add_argument('-output_size', type=int, default=25)                           #Size of chord vector (shouldn't be specified often)
    parser.add_argument('-h_size', type=int, default=128)                               #Size of hidden layer
    parser.add_argument('-n_layers', type=int, default=2)                               #Number of LSTM layers
    parser.add_argument('-noise_size', type=int, default=12)                            #Size of noise vector
    parser.add_argument('-max_seqlen', type=int, default=200)                           #Sequence length limit
    parser.add_argument('-src_data', default='./Data/LSTM/LSTM_data_correct/processed_train')   #Path to dataset
    parser.add_argument('-one_hot', action='store_true')                                #Whether the generator output is converted to one hot or not
    parser.add_argument('-batch_size', type=int, default=10)                            #Batch size
    parser.add_argument('-epochs', type=int, default=100)                               #Number of epochs
    parser.add_argument('-printevery', type=int, default=10)   
    parser.add_argument('-load', action='store_true')                         #How often to print example of progress in number of epochs
    parser.add_argument('-load_dir', type=str, default='./Model/saved_models')
    parser.add_argument('-model_num', type=str, default=1)
    parser.add_argument('-save', action='store_true')
    parser.add_argument('-save_dir', type=str, default='./Model/saved_models')
    # parser.add_argument('-generate', action='store_true')

    opt = parser.parse_args()

    # if opt.generate:
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Preparing Dataset")
    opt.dataset = MusicDataset(opt)
    if opt.load:
        print("Loading Model")
        discriminator = torch.load(f"{opt.load_dir}/discriminators/discriminator_{opt.model_num}.pt")
        generator = torch.load(f"{opt.load_dir}/generators/generator_{opt.model_num}.pt")
    else:
        discriminator = LSTM_Discriminator_Model(opt.device, opt.input_size+opt.output_size, opt.h_size, opt.dataset.max_length, opt.n_layers, 1)
        generator = LSTM_Generator_Model(opt.device ,opt.input_size+opt.noise_size, opt.h_size, opt.n_layers, opt.output_size)
    # print(generator)
    # print(dict(list(generator.named_parameters())))
    # print(discriminator)
    # print(dict(list(discriminator.named_parameters())))
    model = {
        "Generator" : generator,
        "Discriminator" : discriminator
    }
    
    train_model(model, opt)
    # train_LSTM(opt)

if __name__ == "__main__":
    main()
