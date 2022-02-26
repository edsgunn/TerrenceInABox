from Piano import Piano
import pandas as pd
from gensound import Signal, Sine

sig = Piano

beat = 1.25e2 # 120 bpm


s = Signal()
f = pd.read_csv("Model/dataset/csv_train/Here Comes the Sun.csv")
print(f[["note_root","note_duration"]])
nums = "0123456789"
for index, row in f.iterrows():
    note = row["note_root"]
    for char in nums:
        note = note.replace(char,"")
    note = note.replace("rest","r")
    duration = row["note_duration"]
    s |= Piano(note, duration=beat*duration)

s.play()