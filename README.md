# Tch(ai)kovsky

[Demo of generated music after 600 epochs](https://github.com/NickTikhonov/tchaikovsky/releases/download/0.1.0/sample.mp3)

A layman's attempt at generating music with LSTMs :) 

The model is able to train on a large corpus of MIDI data and produce some reasonable sounding MIDI.
There is a focus on representing data in a way to allow the generation of chords, and learning of harmony.

## Setup
* Throw in some midi files into `/data`
* Tune `main.py` to configure number of epochs
* Run the training - note, this is configured to train on an Nvidia GPU - I used an RTX 2070S
* Midi output can be generated using `generate()`

## Data processing / representation
The internal representaiton of music is a sequence of chords:
* Chords have root notes (lowest note in the chord) and harmonics (other notes played together with the root note)
* Root notes are represented by its midi value (semitone count in the range 0-127). Specifically, root notes are represented by thes semitone offset from the prior root note. This allows a piece of music to be transposed up and down (change in key) and have the exact same internal representation. This is a nice property to have, as it allows the network to learn harmonic progressions via examples presented in different keys!
* Chords are represented by their shape - distance of each note from the root note, e.g. a major chord is [0, 4, 7]
* The preprocessing routine simplifies chords, constraining them to a maximum size - e.g. `max_chord=3` ensures that a chord cannot have more than 3 notes. It also imposes a constraint on the maximum semitone distance of each harmonic - `max_dist=11` allows for [0, 5, 11] but not [0, 12]. These constraints allow any chord to be represented with a constant number of bits (which seems useful for training a neural network).

## TODO:
- [ ] Figure out how to train a model with multiple categorical outputs.
- [ ] Figure out how to represent / generate note lengths + delays
- [ ] Encode rhythmic structure into the model



