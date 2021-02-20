
import numpy as np
from numpy import load
from keras.models import load_model
from sklearn.model_selection import train_test_split

notes_array = load('C:/Users/Jagadish/Desktop/images/data.npy', allow_pickle=True)
# No. of unique notes
notes_ = [element for note_ in notes_array for element in note_]

unique_notes = list(set(notes_))
print(len(unique_notes))

# importing library
from collections import Counter

# computing frequency of each note
freq = dict(Counter(notes_))

# library for visualiation
# import matplotlib.pyplot as plt

# consider only the frequencies
no = [count for _, count in freq.items()]

# set the figure size
frequent_notes = [note_ for note_, count in freq.items() if count >= 50]
# print(len(frequent_notes))

new_music = []

for notes in notes_array:
    temp = []
    for note_ in notes:
        if note_ in frequent_notes:
            temp.append(note_)
    new_music.append(temp)

new_music = np.array(new_music)
no_of_timesteps = 32
x = []
y = []

for note_ in new_music:
    for i in range(0, len(note_) - no_of_timesteps, 1):
        # preparing input and output sequences
        input_ = note_[i:i + no_of_timesteps]
        output = note_[i + no_of_timesteps]

        x.append(input_)
        y.append(output)

x = np.array(x)
y = np.array(y)

unique_x = list(set(x.ravel()))
x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))

# preparing input sequences
x_seq = []
for i in x:
    temp = []
    for j in i:
        # assigning unique integer to every note
        temp.append(x_note_to_int[j])
    x_seq.append(temp)

x_seq = np.array(x_seq)

unique_y = list(set(y))
y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
y_seq = np.array([y_note_to_int[i] for i in y])

x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

model = load_model('C:/Users/Jagadish/Desktop/images/my_model')

def generate(painoList):
    random_music =painoList

    random_music = np.array(random_music)
    # print((random_music))

    predictions = []
    for i in range(12):
        random_music = random_music.reshape(1, no_of_timesteps)

        prob = model.predict(random_music)[0]
        y_pred = np.argmax(prob, axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0], len(random_music[0]), y_pred)
        random_music = random_music[1:]

    x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))
    predicted_notes = [x_int_to_note[i] for i in predictions]
    # print((predicted_notes))
    return predicted_notes


