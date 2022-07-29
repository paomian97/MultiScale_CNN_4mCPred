import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, Flatten, Dense, Dropout, Bidirectional
import numpy as np

def numerical_transform(sequences):
    unit=[x for x in 'AGCT']
    Dict={Key:value for value, Key in enumerate(unit)}
    outcome=[]
    for seq in sequences:
        outcome.append([Dict[a] for a in seq])
    return np.array(outcome)

def model_framework():
    x = tf.keras.Input(shape=(41))

    embedding = Embedding(input_dim=4, output_dim=8)(x)

    lstm_1 = Bidirectional(LSTM(6, return_sequences=True))(embedding)

    conv_1 = Conv1D(filters=9, kernel_size=3, strides=1, activation='relu')(lstm_1)
    drop_conv_1 = Dropout(0.3)(conv_1)

    conv_2 = Conv1D(filters=9, kernel_size=5, strides=1, activation='relu')(lstm_1)
    drop_conv_2 = Dropout(0.3)(conv_2)

    conv_3 = Conv1D(filters=9, kernel_size=7, strides=1, activation='relu')(lstm_1)
    drop_conv_3 = Dropout(0.3)(conv_3)

    concate = tf.concat([drop_conv_1, drop_conv_2, drop_conv_3], axis=1)

    flatten = Flatten()(concate)

    dense_1 = Dense(27, activation='relu')(flatten)
    drop = Dropout(0.3)(dense_1)
    dense_2 = Dense(9, activation='relu')(drop)
    dense_3 = Dense(1, activation='sigmoid')(dense_2)

    y = dense_3

    model = tf.keras.Model(inputs=x, outputs=y)
    model.compile(optimizer='Adam',
                  loss='BinaryCrossentropy',
                  metrics='accuracy')

    return model

def train_model(train_x, train_y):
    train_feature = numerical_transform(train_x)
    model = model_framework()
    model.fit(x=train_feature, y=train_y, batch_size=8, epoch=100)

    #model.save('./model_save')