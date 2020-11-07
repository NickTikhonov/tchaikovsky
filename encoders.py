import numpy as np
from data_model import Piece, Note

# Maximum pause for a note in quarters.
MAX_NOTE_PAUSE = 4.0

class AlphaEncoder:
    @staticmethod
    def note_to_x(note, prev_note = None, max_chord = 3, max_dist = 24):
        chord = [x / max_dist for x in note.chord_shape(max_chord=max_chord, max_dist=max_dist)]
        offset = 0
        pause = 0.5
        if prev_note:
            offset = note.root() - prev_note.root()
            pause = note.t - prev_note.t
            assert pause != 0
            if pause > MAX_NOTE_PAUSE:
                pause = MAX_NOTE_PAUSE
        offset += 127
        out = [offset / 255, pause / MAX_NOTE_PAUSE]
        out.extend(chord[1:])
        return out

    @staticmethod
    def note_from_x(x, prev_note = None, max_chord = 3, max_dist = 24):
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
    @staticmethod
    def note_to_y(note, prev_note = None, max_chord = 3, max_dist = 24):
        chord = note.chord_shape(max_chord=max_chord, max_dist=max_dist)
        offset = 0
        if prev_note:
            offset = note.root() - prev_note.root()
        offset += 127
        out = np.eye(255)[offset].tolist()
        for c in np.eye(max_dist)[chord[1:]]:
            out.extend(c.tolist())
        return np.asarray(out)

    @staticmethod
    def note_from_y(y, prev_note = None, max_chord = 3, max_dist = 24):
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

    @staticmethod
    def encode(piece, seq_len=100, max_chord=3, max_dist=24):
        if len(piece.notes) <= seq_len:
            return [], []

        xs = []
        ys = []
        for i in range(len(piece.notes) - 1):
            xs.append(AlphaEncoder.note_to_x(piece.notes[i + 1], prev_note=piece.notes[i], max_chord=max_chord, max_dist=max_dist))
            ys.append(AlphaEncoder.note_to_y(piece.notes[i + 1], prev_note=piece.notes[i], max_chord=max_chord, max_dist=max_dist))

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
    def decode(xs, max_chord=3, max_dist=24):
        p = Piece()
        prev_note = None
        for i in range(0, len(xs) - 1):
            n = AlphaEncoder.note_from_x(xs[i], prev_note=prev_note, max_chord=max_chord, max_dist=max_dist)
            p.add(n)
            prev_note = n
        return p



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

