import datetime
import pathlib

import numpy as np
import glob
from tqdm import tqdm

import tensorflow as tf

config = tf.compat.v1.ConfigProto()

# pylint: disable=no-memeber
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

from keras_self_attention import SeqSelfAttention

from data_model import Piece, Note 
from encoders import AlphaEncoder
        

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

def generate_piece(model, seed_seq, num_notes):
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

    return AlphaEncoder.decode(music, max_chord=5, max_dist=24)

if __name__ == "__main__":
    train_x = []
    train_y = []
    for fname in tqdm(glob.glob("data/**/*.mid")[0:10]):
        p = Piece.from_midi(fname)
        xs, ys = AlphaEncoder.encode(p, seq_len=100, max_chord=5, max_dist=24)
        train_x.extend(xs)
        train_y.extend(ys)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    with open('data/train_x.npy', 'wb') as f:
         np.save(f, train_x)
    with open('data/train_y.npy', 'wb') as f:
         np.save(f, train_y)

    train_x = np.load('data/train_x.npy')
    print(train_x[0])
    train_y = np.load('data/train_y.npy')
    print(train_y[0])

    print("Building model")
    model = build_model((train_x.shape[1], train_x.shape[2]), train_y.shape[1])

    print("Loading weights")
    model.load_weights('models/weights-improvement-63-0.0071-bigger.hdf5')
    train(model, train_x, train_y, epochs=1)

    print("Generating music")
    for i in range(10):
        generate_piece(model, train_x[i*5], 150).to_midi(f"output/with-self-attention-{i}.midi")
