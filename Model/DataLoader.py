from torch.utils.data import Dataset
import glob
import pandas as pd
import torch
# import matplotlib.pyplot as plt

class MusicDataset(Dataset):
    def __init__(self):
        self.chords = []
        self.melody = []
        lengths = []
        data_dir = "./transformed_dataset/processed_train"
        for file in glob.glob(f"{data_dir}/*.csv"):
            f = pd.read_csv(file)
            melody = f.iloc[:,[i for i in range(12)]].values
            melody = torch.tensor(melody)
            chord = f.iloc[:,[i for i in range(12,36)]].values
            chord = torch.tensor(chord)
            length = f.shape[0]
            if length < 400:
                self.chords.append(chord)
                self.melody.append(melody)
                lengths.append(f.shape[0])
        max_length = max(lengths)
        for i in range(len(self.chords)):
            diff = max_length-len(self.chords[i])
            m = torch.nn.ConstantPad2d((0, 0, 0, diff), 0)
            self.chords[i] = m(self.chords[i]).float()
            self.melody[i] = m(self.melody[i]).float()
        # print(max([l for l in lengths if l < 613]))
        # print(lengths.index(max_length))
        # plt.hist(lengths, bins = [i for i in range(849)])
        # plt.show()

    def __len__(self):
        return len(self.chords)

    def __getitem__(self, idx):
        label = self.chords[idx]
        data = self.melody[idx]
        sample = {"Melody": data, "Chords": label}
        return sample

# training_parameters = {
#     "n_epochs": 100,
#     "batch_size": 10,
# }
# data_loader = torch.utils.data.DataLoader(MusicDataset(),batch_size=training_parameters["batch_size"], shuffle=True)

# for epoch_idx in range(training_parameters["n_epochs"]):
#     G_loss = []
#     D_loss = []
#     for batch_idx, data_input in enumerate(data_loader):
#       print(data_input["Chords"])
#       break
#     break