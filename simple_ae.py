import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras

print(tf.__version__)


### DATA
training_data = np.random.rand(1000, 90)
train_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_data))
train_dataset = train_dataset.shuffle(1000).batch(100)

### MODEL
x = keras.layers.Input(shape=(90,))
h = keras.layers.Dense(40, activation=tf.nn.relu)(x)
z = keras.layers.Dense(10, activation=tf.nn.relu)(h)
h_decoded = keras.layers.Dense(40, activation=tf.nn.relu)(z)
x_decoded = keras.layers.Dense(90)(h_decoded)

model = keras.models.Model(x, x_decoded)

### LOSS
recon_err = tf.reduce_sum(tf.abs(x - x_decoded), axis=1)
total_loss = tf.reduce_mean(recon_err)
model.add_loss(total_loss)

### TRAINING
model.compile(optimizer='adam')
print(model.summary())
model.fit(train_dataset, epochs=5)

### CONVERSION
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_file_name = 'simple_ae.tflite'
tflite_model = converter.convert()
open(tflite_file_name, 'wb').write(tflite_model)


#################################################################################################

interpreter = tf.lite.Interpreter(model_path=tflite_file_name)
interpreter.allocate_tensors()
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]
print('Input detail: ', input_detail)
print('Output detail: ', output_detail)

input_data = np.random.rand(1, 90).astype(np.float32)
interpreter.set_tensor(input_detail['index'], input_data)
interpreter.invoke()
pred_litemodel = interpreter.get_tensor(output_detail['index'])

print(f'\nPredicted output')
for prediction in pred_litemodel[0]:
	print(prediction)
print()