from Piano import Piano
import pandas as pd
from gensound import Signal, Sine
import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import os
sys.path.append(f'{os.getcwd()}/Model')
from Discriminator import LSTM_Discriminator_Model
from Generator import LSTM_Generator_Model

def constrainNote(note):
    nums = "0123456789"
    for char in nums:
        note = note.replace(char,"")
    # note = note.replace("rest","r")
    return note
def oneHotToChord(chords):
    chord_map= [("C","major"),("C#","major"),("D","major"),("D#","major"),("E","major"),("F","major"),("F#","major"),("G","major"),("G#","major"),("A","major"),("A#","major"),("B","major"),("C","minor"),("C#","minor"),("D","minor"),("D#","minor"),("E","minor"),("F","minor"),("F#","minor"),("G","minor"),("G#","minor"),("A","minor"),("A#","minor"),("B","minor"),"rest"]
    transformed_chords=[chord_map[np.argmax(chord)] for chord in chords.squeeze()]
    return transformed_chords

def generateChord(melody,model):
    noise = torch.randn(1,183,12)
    noise = torch.cat((noise,melody.view(1,183,12)),2)
    chord = model(noise.to('cuda:0'))
    return oneHotToChord(chord.to('cpu').detach().numpy())

def normaliseSong(song):
    # Define all mappings and major keys

    # Mapping between notes and numerical representation
    note_map = {
        'C0':1,
        'B#':1,
        'C#':2,
        'Db':2,
        'D0':3,
        'D#':4,
        'Eb':4,
        'E0':5,
        'Fb':5,
        'F0':6,
        'E#':6,
        'F#':7,
        'Gb':7,
        'G0':8,
        'G#':9,
        'Ab':9,
        'A0':10,
        'A#':11,
        'Bb':11,
        'B0':12,
        'Cb':12
    }

    # Mapping between major key and notes
    C_sharp = [2,4,6,7,9,11,1]
    F_sharp = [7,9,11,12,2,4,6]
    B = [12,2,4,5,7,9,11]
    E = [5,7,9,10,12,2,4]
    A = [10,12,2,3,5,7,9]
    D = [3,5,7,8,10,12,2]
    G = [8,10,12,1,3,5,7]
    C = [1,3,5,6,8,10,12]
    F = [6,8,10,11,1,3,5]
    B_flat = [11,1,3,4,6,8,10]
    E_flat = [4,6,8,9,11,1,3]
    A_flat = [9,11,1,2,4,6,8]
    D_flat = [2,4,6,7,9,11,1]
    G_flat = [7,9,11,12,2,4,6]

# Key_fifths mapping to major key
    kf_map = {
        -6:G_flat,
        -5:D_flat,
        -4:A_flat,
        -3:E_flat,
        -2:B_flat,
        -1:F,
        0:C,
        1:G,
        2:D,
        3:A,
        4:E,
        5:B,
        6:F_sharp,
        7:C_sharp
    }

    time_map = {
    '4/4':1,
    '3/4':4/3,
    '2/2':1,
    '6/8':8/6,
    '2/4':2,
    '12/8':8/12,
    '6/4':4/6,
    '9/8':8/9,
    '3/8':8/3,
    '5/4':4/5,
    '1/2':2,
    '4/8':2
    }

    chord_map = {
    'major':'major',
    'dominant':'major',
    'minor':'minor',
    'minor-seventh':'minor',
    'major-seventh':'major',
    'maj':'major',
    'major-sixth':'major',
    'dominant-ninth':'major',
    'min':'minor',
    'minor-sixth':'minor',
    '7':'major',
    'suspended-fourth':'major',
    'diminished':'minor',
    'half-diminished':'minor',
    'minor-ninth':'minor',
    'diminished-seventh':'minor',
    'augmented-seventh':'major',
    'min7':'minor',
    'major-ninth':'major',
    'maj7':'major',
    'dominant-seventh':'major',
    'augmented': 'major',
    'dominant-13th':'major',
    'power':'major',
    'suspended-second':'major',
    'dominant-11th':'major',
    'dim':'minor',
    'minor-11th':'minor',
    'minor-major':'minor',
    'major-minor':'minor',
    'maj9':'major',
    '9':'major',
    'pedal':'major',
    'maj69':'major',
    'aug':'major',
    'min9':'minor',
    'augmented-ninth':'major',
    'minor-13th':'minor',
    '6':'major',
    'm7b5':'minor',
    'minMaj7':'minor',
    'sus47':'major',
    'dim7':'minor',
    ' dim7':'minor'
    }
    
    data = song.loc[~song['note_root'].isin(['rest','F2','B-2','C2','D-2','A2'])]
    #data = data.loc[~data['chord_root'].isin(['[]'])]
    #data = data.loc[~data['chord_type'].isin(['[]'])]
    data['chord_root'] = data['chord_root'].fillna('[]')
    data['chord_type'] = data['chord_type'].fillna('[]')
    data = data.dropna()  # Remove NULL values
    data.drop('note_octave', inplace=True, axis=1)  # Remove octave information

    li2 = data.to_numpy()

    # Shift rootnote and rootchord to C major key, and convert chord type to either major or minor
    n = len(li2)
    shifted_li = li2.copy()

    for i in range(n):
        row = li2[i]
        rootnote = row[6]
        notenum = note_map[rootnote]
        kf = row[2]
        major = kf_map[kf]

        rootchord = row[4]
        chordnum = note_map[rootchord]
        chordtype = row[5] # Chord type, eg. major, diminished

        time = row[0]
        normalised_time = time_map[time]
        note_duration = row[7]

        # Find the index of the number in major closest to notenum, but this isnt necessary
        #index = min(range(7), key = lambda j:abs(major[j]-notenum))

        # Find the difference between notenum and the major key num, and add it to the C major key num to get the shifted note.
        # This works since the intervals between each note in a major key is the same across all major keys
        difference = notenum-major[0]
        shifted_note = C[0] + difference

        if shifted_note <=0:
            shifted_note = shifted_note + 12
        elif shifted_note>12:
            shifted_note = shifted_note - 12

        shifted_li[i,6] = shifted_note

        
        # Check if the chord exists, i.e. 'rest chord'
        if chordnum != '[]':
            difference = chordnum-major[0]
            shifted_chord = C[0] + difference

            if shifted_chord <=0:
                shifted_chord = shifted_chord + 12
            elif shifted_chord>12:
                shifted_chord = shifted_chord - 12
        else:
            shifted_chord = '[]'

        # Check if there is a chord type
        if chordtype != '[]':
            shifted_li[i,5] = chord_map[chordtype]
        else:
            shifted_li[i,5] = '[]'
            
        shifted_li[i,4] = shifted_chord
        shifted_li[i,7] = normalised_time*note_duration
        
        
    new_data = []
    start = 1
    measure = 'unknown'
    new_row = np.zeros((1,37))
    chord_saved = 0
    for i in range(n):
        row = shifted_li[i,:]
        rootnote = row[6]
        rootchord = row[4]
        chordtype = row[5]
        normalised_note_duration = row[7]

        if measure == row[1]:
            new_row[0,rootnote-1] +=normalised_note_duration

        else:  # New measure

            # First measure of the song
            if start == 1:
                #new_data = np.zeros((1,37))
                new_row[0,rootnote-1] += normalised_note_duration
                measure = row[1]
                chord_saved = chordtype

                if chordtype == 'major':
                    new_row[0,rootchord+11] = 1
                elif chordtype == '[]':
                    new_row[0,-1] = 1
                else:
                    new_row[0,rootchord+23] = 1
                start = 0

            # Transition from one measure to the next
            else:
                # Add the row for the previous measure to new_data
                if len(new_data) == 0:
                    new_data = new_row
                else:
                    new_data = np.concatenate((new_data, new_row), axis=0)

                # Update the measure to current measure, reset new_row
                measure = row[1]
                new_row = np.zeros((1,37))


                new_row[0,rootnote-1] += normalised_note_duration

                if chordtype == 'major':
                    new_row[0,rootchord+11] = 1
                elif chordtype == '[]':
                    new_row[0,-1] = 1
                else:
                    new_row[0,rootchord+23] = 1





    df = pd.DataFrame(new_data, index=None, columns=["C", "C#",'D','D#','E','F','F#','G','G#','A','A#','B',"C major", "C# major",'D minor','D# major','E major','F major','F# major','G major','G# major','A major','A# major','B major',"C minor", "C# minor",'D minor','D# minor','E minor','F minor','F# minor','G minor','G# minor','A minor','A# minor','B minor','No chords'])
    return df
        

def generateChords(model,sequence_length,song):
    normalisedSong = normaliseSong(song)
    normalisedMelody = normalisedSong.iloc[:,[i for i in range(12)]].values
    normalisedMelody = torch.tensor(normalisedMelody)
    l = len(normalisedMelody)
    diff = sequence_length-l
    m = nn.ConstantPad2d((0, 0, 0, diff), 0)
    normalisedMelody = m(normalisedMelody).float()
    return generateChord(normalisedMelody,model)

def playChords(melody,chords):
    chord_map = {
    'major':'major',
    'dominant':'major',
    'minor':'minor',
    'minor-seventh':'minor',
    'major-seventh':'major',
    'maj':'major',
    'major-sixth':'major',
    'dominant-ninth':'major',
    'min':'minor',
    'minor-sixth':'minor',
    '7':'major',
    'suspended-fourth':'major',
    'diminished':'minor',
    'half-diminished':'minor',
    'minor-ninth':'minor',
    'diminished-seventh':'minor',
    'augmented-seventh':'major',
    'min7':'minor',
    'major-ninth':'major',
    'maj7':'major',
    'dominant-seventh':'major',
    'augmented': 'major',
    'dominant-13th':'major',
    'power':'major',
    'suspended-second':'major',
    'dominant-11th':'major',
    'dim':'minor',
    'minor-11th':'minor',
    'minor-major':'minor',
    'major-minor':'minor',
    'maj9':'major',
    '9':'major',
    'pedal':'major',
    'maj69':'major',
    'aug':'major',
    'min9':'minor',
    'augmented-ninth':'major',
    'minor-13th':'minor',
    '6':'major',
    'm7b5':'minor',
    'minMaj7':'minor',
    'sus47':'major',
    'dim7':'minor',
    ' dim7':'minor'
    }


    major = {"C":["C","E","G"],
            "C#":["C#","E#","G#"],
            "D":["D","F#","A"],
            "D#":["Eb","G","Bb"],
            "Eb":["Eb","G","Bb"],
            "E":["E","G#","B"],
            "F":["F","A","C"],
            "F#":["F#","A#","C#"],
            "G":["G","B","D"],
            "G#":["Ab","C","Eb"],
            "Ab":["Ab","C","Eb"],
            "A":["A","C#","E"],
            "A#":["A#","D", "F"],
            "Bb":["Bb","D","F"],
            "B":["B","D#","F#"]
    }
    minor = {"C":["C","Eb","G"],
            "C#":["C#","E","G#"],
            "D":["D","F","A"],
            "D#":["Eb","Gb","Bb"],
            "Eb":["Eb","Gb","Bb"],
            "E":["E","G","B"],
            "F":["F","Ab","C"],
            "F#":["F#","A","C#"],
            "G":["G","Bb","D"],
            "G#":["Ab","Cb","Eb"],
            "Ab":["Ab","Cb","Eb"],
            "A":["A","C","E"],
            "A#":["A#","C#","F"],
            "Bb":["Bb","Db","F"],
            "B":["B","D","F#"]
    }


    chord_types = {"major":major,"minor":minor}
    sig = Piano

    beat = 1.25e2 # 120 bpm

    time_sig = melody["time"][0]
    beats = int(time_sig[0])
    semiquavers = 16/int(time_sig[2])
    measure = beats*semiquavers
    tune = ""
    chord1 = ""
    chord2 = ""
    chord3 = ""
    current_measure = 0
    
    for index, row in melody.iterrows():
        note = row["note_root"]
        duration = row["note_duration"]
        
        if note == "rest":
            tune += f"r={duration} "
        else:
            note = constrainNote(note)
            tune += f"{note}4={duration} "
        if row["measure"] != current_measure:
            current_measure = row["measure"]
            chord = chords[current_measure]
            if chord == "rest":
                chord1 += f"r={measure} "
                chord2 += f"r={measure} "
                chord3 += f"r={measure} "
            else:
                chord_root = constrainNote(chord[0])
                key_mode = chord[1]
                chord = chord_types[key_mode][chord_root]
                # print(chord)
                chord1 += f"{chord[0]}4={measure} "
                chord2 += f"{chord[1]}4={measure} "
                chord3 += f"{chord[2]}4={measure} "
            
    print(tune)
    print(chord1)
    print(chord2)
    print(chord3)
    m = sig(tune, duration=beat)
    c1 = sig(chord1, duration=beat)
    c2 = sig(chord2, duration=beat)
    c3 = sig(chord3, duration=beat)
    s = m + c1 + c2 + c3

    s.play(max_amplitude=1)
    print("Finished")


def accompanySong(model,songname):
    melody = pd.read_csv(f"Model/dataset/csv_train/{songname}.csv")
    chords = generateChords(model,183,melody)
    print(chords)
    playChords(melody,chords)

def main():

    parser = argparse.ArgumentParser()   
    parser.add_argument('-songname', type=str, default="DON'T STOP BELIEVIN'")                    
    parser.add_argument('-load_dir', type=str, default='./Model/saved_models')
    parser.add_argument('-model_num', type=str, default=1)

    opt = parser.parse_args()
    model = torch.load(f"{opt.load_dir}/generators/generator_{opt.model_num}.pt")
    accompanySong(model,opt.songname)

if __name__ == "__main__":
    main()
    