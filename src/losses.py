import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow import map_fn, argmax, gather
from distributions import get_beta_probabilities, get_poisson_probabilities, get_binominal_probabilities, \
	get_exponential_probabilities

# Compute categorical cross-entropy applying regularization based on beta distribution to targets.
def categorical_ce_beta_regularized(num_classes, eta=1.0):
	# Params [a,b] for beta distribution
	params = {}

	params['4'] = [
		[1,6],
		[6,10],
		[9,6],
		[6,1]
	]

	params['5'] = [
		[1,9], # [2, 18], # [1, 9],
		[6,14], # [6, 14], # [3, 7],
		[12,12], # [10, 10], # [5, 5],
		[14,6], # [14, 6], # [7, 3],
		[9,1] # [18, 2] # [9, 1]
	]
	params['6'] = [
		[1,10],
		[7,20],
		[15,20],
		[20,15],
		[20,7],
		[10,1]
	]

	params['8'] = [
		[1,14],
		[7,31],
		[17,37],
		[27,35],
		[35,27],
		[37,17],
		[31,7],
		[14,1]
	]

	# Precompute class probabilities for each label
	cls_probs = []
	for i in range(0, num_classes):
		cls_probs.append(get_beta_probabilities(num_classes, params[str(num_classes)][i][0], params[str(num_classes)][i][1]))

	def _compute(y_true, y_pred):
		y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
		y_true = (1 - eta) * y_true + eta * y_prob

		return categorical_crossentropy(y_true, y_pred)

	return _compute


# Compute categorical cross-entropy applying regularization based on poisson distribution to targets.
def categorical_ce_poisson_regularized(num_classes, eta=1.0):
	cls_probs = get_poisson_probabilities(num_classes)

	def _compute(y_true, y_pred):
		y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
		y_true = (1 - eta) * y_true + eta * y_prob

		return categorical_crossentropy(y_true, y_pred)

	return _compute

# Compute categorical cross-entropy applying regularization based on binomial distribution to targets.
def categorical_ce_binomial_regularized(num_classes, eta=1.0):
	cls_probs = get_binominal_probabilities(num_classes)

	def _compute(y_true, y_pred):
		y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
		y_true = (1 - eta) * y_true + eta * y_prob

		return categorical_crossentropy(y_true, y_pred)

	return _compute


# Compute categorical cross-entropy applying regularization based on exponential distribution to targets.
def categorical_ce_exponential_regularized(num_classes, eta=1.0, tau=1.0):
	cls_probs = get_exponential_probabilities(num_classes, tau)

	def _compute(y_true, y_pred):
		y_prob = map_fn(lambda y: tf.cast(gather(cls_probs, argmax(y)), tf.float32), y_true)
		y_true = (1 - eta) * y_true + eta * y_prob

		return categorical_crossentropy(y_true, y_pred)

	return _compute
