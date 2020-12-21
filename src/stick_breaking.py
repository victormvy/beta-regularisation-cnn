import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.regularizers import l2

def get_stick_probabilities(t):
	t = tf.clip_by_value(t, 0.1, 0.9)

	comp = 1.0 - t
	cumprod = tf.cumprod(comp, axis=1)
	cumprod = tf.pad(cumprod, [(0, 0), (1, 0)], mode='CONSTANT', constant_values=1.0)
	t = tf.pad(t, [(0, 0), (0, 1)], mode='CONSTANT', constant_values=1.0)
	return t * cumprod


def stick_breaking_layers(l_in, num_classes):
	layer = Dense(num_classes - 1, activation='sigmoid', kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4))(l_in)
	layer = Lambda(get_stick_probabilities)(layer)
	return layer