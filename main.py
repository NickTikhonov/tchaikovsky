from collections import defaultdict
import datetime
import pathlib

import numpy as np
import glob
from tqdm import tqdm
from music21 import converter, instrument, note, chord, stream

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

from keras_self_attention import SeqSelfAttention

# Maximum pause for a note in quarters.
MAX_NOTE_PAUSE = 4.0

class Piece:
    def get_sequences(self, seq_len = 100, max_chord = 3, max_dist = 24):
        if len(self.notes) <= seq_len:
            return [], []

        xs = []
        ys = []
        for i in range(len(self.notes) - 1):
            xs.append(self.notes[i + 1].to_x(prev_note=self.notes[i], max_chord=max_chord, max_dist=max_dist))
            ys.append(self.notes[i + 1].to_y(prev_note=self.notes[i], max_chord=max_chord, max_dist=max_dist))

        n_in = []
        n_out = []
        for i in range(0, len(xs) - seq_len, 1):
            sequence_in = xs[i:i + seq_len]
            # TODO(Nick): Swap this out to use the value
            # from 'ys' isntead, once there is a way 
            # to trian with multiple categorical outputs.
            sequence_out = xs[i + seq_len]
            n_in.append(sequence_in)
            n_out.append(sequence_out)

        return n_in, n_out
        

    @staticmethod
    def from_xs(xs, max_chord=3, max_dist=24):
        p = Piece()
        prev_note = None
        for i in range(0, len(xs) - 1):
            n = Note.from_x(xs[i], prev_note=prev_note, max_chord=max_chord, max_dist=max_dist)
            print(n.t)
            p.add(n)
            prev_note = n
        return p

    @staticmethod
    def from_midi(fname):
        midi = converter.parse(fname)
        notes = None
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
            notes = parts.parts[0].recurse()
        else: # file has notes in a flat structure
            notes = midi.flat.notes

        # Remove weird crap like cloned timestep notes and other shit
        offset_to_pitches = defaultdict(set)
        for each in notes:
            if isinstance(each, note.Note):
                offset_to_pitches[each.offset].add(each.pitch.ps)
            elif isinstance(each, chord.Chord):
                for n in each.notes:
                    offset_to_pitches[each.offset].add(n.pitch.ps)
            else:
                continue

        p = Piece()
        for offset in sorted(list(offset_to_pitches.keys())):
            pitches = sorted(list(offset_to_pitches[offset]))
            n = Note(offset, pitches[0])
            for pitch in pitches:
                n.add(pitch)
            p.add(n)

        return p

    def to_midi(self, out_path):
        output_notes = []
        for n in self.notes:
            if n.is_chord():
                notes = []
                for pitch in n.pitches():
                    new_note = note.Note(pitch)
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = n.t
                output_notes.append(new_chord)
            else:
                new_note = note.Note(n.root())
                new_note.offset = n.t
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=out_path)

    def __init__(self):
        self.notes = []

    def add(self, note):
        if len(self.notes) > 0:
            prev = self.notes[-1]
            assert prev.t < note.t
        self.notes.append(note)


class Note:
    # Midi pitches are in values of 0-127.
    def __init__(self, t, midi_pitch):
        if midi_pitch < 0:
            midi_pitch = 0
        if midi_pitch > 127:
            midi_pitch = 127
        self.t = t
        self.mp = set([int(midi_pitch)])

    def root(self):
        return sorted(list(self.mp))[0]

    def pitches(self):
        return sorted(list(self.mp))
    
    def is_chord(self):
        return len(self.mp) > 1

    def add(self, midi_pitch):
        if midi_pitch < 0:
            midi_pitch = 0
        if midi_pitch > 127:
            midi_pitch = 127
        self.mp.add(int(midi_pitch))

    def chord_shape(self, max_chord = 3, max_dist = 24):
        ordered = sorted(list(self.mp))
        normed = [x - ordered[0] for x in ordered]
        filtered = list(filter(lambda x: x < max_dist, normed))
        return filtered[0:max_chord] + [0 for _ in range(max_chord - len(filtered))]

    # Returns a vector containing [root_offset, pause, ...chord_shape]
    def to_x(self, prev_note = None, max_chord = 3, max_dist = 24):
        chord = [x / max_dist for x in self.chord_shape(max_chord=max_chord, max_dist=max_dist)]
        offset = 0
        pause = 0.5
        if prev_note:
            offset = self.root() - prev_note.root()
            pause = self.t - prev_note.t
            assert pause != 0
            if pause > MAX_NOTE_PAUSE:
                pause = MAX_NOTE_PAUSE
        offset += 127
        out = [offset / 255, pause / MAX_NOTE_PAUSE]
        out.extend(chord[1:])
        return out

    @staticmethod
    def from_x(x, prev_note = None, max_chord = 3, max_dist = 24):
        x = x.tolist()
        offset = float(int((x[0] * 255) - 127))
        if prev_note:
            offset += prev_note.root()
        
        t = x[1] * MAX_NOTE_PAUSE
        if t <= 0:
            t = 0.1
        chords = x[2:]
        scaled = [offset + int(c * max_dist) for c in chords]

        if prev_note:
            t += prev_note.t 
        n = Note(t, offset)
        for pitch in scaled:
            n.add(pitch)
        return n
    
    # Similar to 'to_x' - but instead encodes into a flattened vector of one-hot vectors
    # representing root note offset and each of the chord offsets
    def to_y(self, prev_note = None, max_chord = 3, max_dist = 24):
        chord = self.chord_shape(max_chord=max_chord, max_dist=max_dist)
        offset = 0
        if prev_note:
            offset = self.root() - prev_note.root()
        offset += 127
        out = np.eye(255)[offset].tolist()
        for c in np.eye(max_dist)[chord[1:]]:
            out.extend(c.tolist())
        return np.asarray(out)

    @staticmethod
    def from_y(y, prev_note = None, max_chord = 3, max_dist = 24):
        offset = np.argmax(y[0:255]) - 127
        chords = []
        for i in range(max_chord - 1):
            chords.append(np.argmax(y[255 + (i * max_dist): 255 + ((i + 1) * max_dist)]))
        t = 0
        if prev_note:
            t = prev_note.t + 1
            offset += prev_note.root()
        n = Note(t, offset)
        for c in chords:
            n.add(offset + c)
        return n

    def __repr__(self):
        return f"<Note {self.t} {[x for x in sorted(list(self.mp))]}>"
        

class ListEncoder:
    def __init__(self, things):
        thing_set = set(item for item in things)
        self.n = len(thing_set)
        self.map = dict((thing, number) for number, thing in enumerate(thing_set))
        self.unmap = dict((number, thing) for thing, number in self.map.items())

    def encode(self, thing):
        return self.map[thing]

    def decode(self, num):
        return self.unmap[num]

    def size(self):
        return self.n


def build_model(input_shape, out_size):
    print("Step: Building Model...")
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(512,
        input_shape=input_shape, 
        return_sequences=True))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.LSTM(512))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(out_size))
    model.add(tf.keras.layers.Activation('linear'))
    model.compile(loss='mse', optimizer='rmsprop')

    return model


def train(model, input, output, epochs=1):
    print("Step: Training Model...")
    filepath = "models/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]     
    model.fit(input, output, epochs=epochs, batch_size=64, callbacks=callbacks_list)

def generate(model, seed_seq, num_notes):
    seq = seed_seq
    music = []
    for _ in tqdm(range(200)):
        # TODO: Fix that shape constant
        input = np.reshape(seq, (1, len(seq), 6))
        output = model.predict(input, verbose=0)
        music.append(output[0])
        seq = seq.tolist()
        seq.extend(output.tolist())
        seq = np.array(seq)
        seq = seq[1:len(seq)]

    return Piece.from_xs(music)

if __name__ == "__main__":
    train_x = []
    train_y = []
    for fname in tqdm(glob.glob("data/**/*.mid")[0:10]):
        p = Piece.from_midi(fname)
        xs, ys = p.get_sequences(seq_len=100, max_chord=5, max_dist=24)
        train_x.extend(xs)
        train_y.extend(ys)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # with open('train_x.npy', 'wb') as f:
    #      np.save(f, train_x)
    # with open('train_y.npy', 'wb') as f:
    #      np.save(f, train_y)

    # train_x = np.load('train_x.npy')
    # print(train_x[0])
    # train_y = np.load('train_y.npy')
    # print(train_y[0])

    print("Building model")
    model = build_model((train_x.shape[1], train_x.shape[2]), train_y.shape[1])

    print("Loading weights")
    model.load_weights('models/weights-improvement-63-0.0071-bigger.hdf5')
    # train(model, train_x, train_y, epochs=400)

    print("Generating music")
    for i in range(10):
        generate(model, train_x[i*5], 150).to_midi(f"output/with-self-attention-{i}.midi")
