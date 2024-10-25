import tensorflow as tf
import pandas as pd
import numpy as np

def model_maker(X_train, y_train, X_test, y_test):
    n_timesteps, n_features, n_outputs=X_train.shape[1], X_train.shape[2], y_train.shape[1]

    conv1d_model=tf.keras.Sequential()
    conv1d_model.add(tf.keras.layers.Conv1D(filters=70, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features), padding='same'))
    conv1d_model.add(tf.keras.layers.Conv1D(filters=70, kernel_size=3, activation='relu', padding='same'))
    conv1d_model.add(tf.keras.layers.Dropout(rate=0.5))
    conv1d_model.add(tf.keras.layers.MaxPooling1D(pool_size=1))
    conv1d_model.add(tf.keras.layers.Flatten())
    conv1d_model.add(tf.keras.layers.Dense(150, activation='relu'))
    conv1d_model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))

    conv1d_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    training_history=conv1d_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    _, accuracy=conv1d_model.evaluate(X_test, y_test, batch_size=32)
    return accuracy, conv1d_model

def save_model_and_weights(model, path):
    model.save(f"{path}/1D_CNN_model.h5")
    model.save_weights(f"{path}/1D_CNN_model_weights")