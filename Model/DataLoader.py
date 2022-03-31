from torch.utils.data import Dataset
import torch.nn as nn
import glob
import pandas as pd
import torch
# import matplotlib.pyplot as plt

class MusicDataset(Dataset):
    def __init__(self, opt):
        self.chords = []
        self.melody = []
        lengths = []
        for file in glob.glob(f"{opt.src_data}/*.csv"):
            f = pd.read_csv(file)
            melody = f.iloc[:,[i for i in range(12)]].values
            if melody.size == 0:
                continue
            melody = torch.tensor(melody)
            chord = f.iloc[:,[i for i in range(12,37)]].values
            chord = torch.tensor(chord)
            length = f.shape[0]
            if length < opt.max_seqlen:
                self.chords.append(chord)
                self.melody.append(melody)
                lengths.append(f.shape[0])
        self.max_length = max(lengths)
        for i in range(len(self.chords)):
            l = len(self.chords[i])
            diff = self.max_length-l
            m = nn.ConstantPad2d((0, 0, 0, diff), 0)
            self.chords[i] = m(self.chords[i]).float()
            self.chords[i][l:,-1] = torch.ones(diff)
            self.melody[i] = m(self.melody[i]).float()
        # print(max([l for l in lengths if l < 613]))
        # print(lengths.index(self.max_length))
        # plt.hist(lengths, bins = [i for i in range(self.max_length+1)])
        # plt.show()

    def __len__(self):
        return len(self.chords)

    def __getitem__(self, idx):
        label = self.chords[idx]
        data = self.melody[idx]
        sample = {"Melody": data, "Chords": label}
        return sample
