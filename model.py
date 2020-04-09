import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
from sklearn.model_selection import train_test_split

DATA_DIR = "./data/"

def prepare_datasets(csv_file):
    # Return a dataframe of filenames (mix of center, left, right)
    df = pd.read_csv(DATA_DIR + csv_file)
    # generate 3 rows (center, left, right) for each entry of csv file 
    steering_offset = [0., 0.2, -0.2]
    camera = ['center', 'left', 'right']
    full_df = pd.DataFrame()
    for i in range(3):
        temp_df = df[[camera[i], 'steering']].copy()
        temp_df['steering'] = temp_df['steering'] + steering_offset[i]
        temp_df.rename(columns = {camera[i]:'filename'}, inplace = True)
        full_df = full_df.append(temp_df, ignore_index=True, sort=False)
    full_df['filename'] = full_df['filename'].str.lstrip()
    # split to train and validation set (9:1)
    X = full_df['filename'].values
    y = full_df['steering'].values 
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=32, shuffle=True)
    return X_train, X_valid, y_train, y_valid

def parse_function(filename, steer_angle):
    # Read image from filename
    image_string = tf.io.read_file(DATA_DIR + filename)
    image = tf.io.decode_jpeg(image_string, channels=3)
    return image, steer_angle

def train_preprocess(image, steer_angle):
    # Preprocessing image for training, following this pipeline:
    # remove 'sky' area -> resize to (66,200) -> random flip (vertical) -> convert to YUV -> normalize 
    processed_image = image[55:-20]
    processed_image = tf.image.resize(processed_image, [66, 200], method='bilinear')
    if np.random.rand() < .5: # Randomly perform horizontal flip
        processed_image = tf.image.flip_left_right(processed_image)
        steer_angle = -steer_angle
    processed_image = processed_image / 255            
    processed_image = tf.image.rgb_to_yuv(processed_image)  # y -> [0, 1], u & v -> [-0.5, 0.5]
    processed_image = (processed_image - [0.5, 0., 0.]) * 2 # normalize all channels to [-1, 1] (better for yuv than below)
    #     processed_image = tf.image.per_image_standardization(processed_image)
    return processed_image, steer_angle            

def preprocess(image):
    # Preprocessing image for prediction, similar to train_preprocess but no random flipping
    processed_image = tf.expand_dims(tf.dtypes.cast(image[55:-20], tf.float32), 0)
    processed_image = tf.image.resize(processed_image, [66, 200], method='bilinear')
    processed_image = processed_image / 255            
    processed_image = tf.image.rgb_to_yuv(processed_image)
    processed_image = (processed_image - [0.5, 0., 0.]) * 2 # normalize all channels to [-1, 1]
    return processed_image 

def dataloader(X, y, batch_size=32):
    return tf.data.Dataset.from_tensor_slices((X, y)) \
                          .shuffle(len(X))            \
                          .map(parse_function)        \
                          .map(train_preprocess)      \
                          .batch(batch_size)          \
                          .prefetch(1)

def main():  
    # define model architecture
    PilotNet = models.Sequential([
            # Three Convolution Layers: 5x5, strides = 2
            layers.Conv2D(24, (5, 5), strides=(2,2), activation='relu', input_shape=(66,200,3)),
            layers.Conv2D(36, (5, 5), strides=(2,2), activation='relu'),
            layers.Conv2D(48, (5, 5), strides=(2,2), activation='relu'),
            # Two Convolution Layers: 3x3, strides = 1
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            # Flatten Layer (output=1152)
            layers.Flatten(),
            # Three Fully-connected Layers with Dropout(0.5)
            layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(l=0.001)),
            layers.Dropout(0.5),
            layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l=0.001)),
            layers.Dropout(0.5),
            layers.Dense(10, activation='relu', kernel_regularizer=regularizers.l2(l=0.001)),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])

    # generate dataset (filenames only)
    X_train, X_valid, y_train, y_valid = prepare_datasets('driving_log.csv')
    train_dataloader = dataloader(X_train, y_train)
    valid_dataloader = dataloader(X_valid, y_valid)

    # train model
    optim = tf.keras.optimizers.Adam(learning_rate=0.0003)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    PilotNet.compile(loss="mean_squared_error", optimizer=optim, metrics=["mse"])
    history = PilotNet.fit(train_dataloader, epochs=50, validation_data=valid_dataloader, callbacks=[early_stopping_cb])
    PilotNet.save("model.h5")


if __name__ == '__main__':
    main()