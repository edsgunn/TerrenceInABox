from Piano import Piano
import pandas as pd
from gensound import Signal, Sine

def constrainNote(note):
    nums = "0123456789"
    for char in nums:
        note = note.replace(char,"")
    # note = note.replace("rest","r")
    return note

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

# C major (C). C - E - G
# C# major (C#). C# - E# - G#
# D major (D). D - F# - A
# Eb major (Eb). Eb - G - Bb
# E major (E). E - G# - B
# F major (F). F - A - C
# F# major (F#). F# - A# - C#
# G major (G). G - B - D
# Ab major (Ab). Ab - C - Eb
# A major (A). A - C# - E
# Bb major (Bb). Bb - D - F
# B major (B). B - D# - F#

# C minor (Cm). C - Eb - G
# C# minor (C#m). C# - E - G#
# D minor (Dm). D - F -A
# Eb minor (Ebm). Eb - Gb - Bb
# E minor (Em). E - G - B
# F minor (Fm). F - Ab - C
# F# minor (F#m). F# - A - C#
# G minor (Gm). G - Bb - D
# Ab minor (Abm). Ab - Cb - Eb
# A minor (Am). A - C - E
# Bb minor (Bbm). Bb - Db - F
# B minor (Bm). B - D - F#

major = {"C":["C","E","G"],
        "C#":["C#","E#","G#"],
        "D":["D","F#","A"],
        "Eb":["Eb","G","Bb"],
        "E":["E","G#","B"],
        "F":["F","A","C"],
        "F#":["F#","A#","C#"],
        "G":["G","B","D"],
        "G#":["Ab","C","Eb"],
        "Ab":["Ab","C","Eb"],
        "A":["A","C#","E"],
        "Bb":["Bb","D","F"],
        "B":["B","D#","F#"]
}
minor = {"C":["C","Eb","G"],
        "C#":["C#","E","G#"],
        "D":["D","F","A"],
        "Eb":["Eb","Gb","Bb"],
        "E":["E","G","B"],
        "F":["F","Ab","C"],
        "F#":["F#","A","C#"],
        "G":["G","Bb","D"],
        "G#":["Ab","Cb","Eb"],
        "Ab":["Ab","Cb","Eb"],
        "A":["A","C","E"],
        "Bb":["Bb","Db","F"],
        "B":["B","D","F#"]
}


chords = {"major":major,"minor":minor}


sig = Piano

beat = 1.25e2 # 120 bpm

f = pd.read_csv("Model/dataset/csv_train/Hey soul sister.csv")
# print(f[["note_root","note_duration","chord_root","key_mode"]])
time_sig = f["time"][0]
beats = int(time_sig[0])
semiquavers = 16/int(time_sig[2])
measure = beats*semiquavers
tune = ""
chord1 = ""
chord2 = ""
chord3 = ""
current_meansure = 0
for index, row in f.iterrows():
    note = row["note_root"]
    duration = row["note_duration"]
    
    if note == "rest":
        tune += f"r={duration} "
    else:
        note = constrainNote(note)
        tune += f"{note}4={duration} "
    if row["measure"] != current_meansure:
        chord_root = row["chord_root"]
        if chord_root == "[]" or chord_root == "null":
            chord1 += f"r={measure} "
            chord2 += f"r={measure} "
            chord3 += f"r={measure} "
        else:
            chord_root = constrainNote(chord_root)
            key_mode = chord_map[row["chord_type"]]
            chord = chords[key_mode][chord_root]
            # print(chord)
            chord1 += f"{chord[0]}4={measure} "
            chord2 += f"{chord[1]}4={measure} "
            chord3 += f"{chord[2]}4={measure} "
        current_meansure += 1
print(tune)
print(chord1)
print(chord2)
print(chord3)
m = Piano(tune, duration=beat)
c1 = Piano(chord1, duration=beat)
c2 = Piano(chord2, duration=beat)
c3 = Piano(chord3, duration=beat)
s = m + c1 + c2 + c3

s.play(max_amplitude=1)
print("Finished")