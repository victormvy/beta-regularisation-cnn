# Only import argparse at the begining to speed up execution
import argparse
from pathlib import Path

# Command line arguments processing
parser = argparse.ArgumentParser(description='Experiment settings')
parser.add_argument('results_path', type=Path, help='Path where result files will be stored')
parser.add_argument('output', choices=['stick', 'softmax'], help='Type of output')
parser.add_argument('loss', choices=['ce', 'ce_poisson', 'ce_binomial', 'ce_exponential', 'ce_beta'], help='Loss function')
parser.add_argument('-d', '--dataset', type=str, default='retinopathy', help='Name of the dataset')
parser.add_argument('-n', '--name', type=str, help='Name of the experiment (used in results file)')
parser.add_argument('-f', '--fold', type=int, default=0, help='Start at this fold (used to resume experiments)')
parser.add_argument('-b', '--batch', type=int, default=128, help='Batch size')
parser.add_argument('-r', '--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

# Import the rest of the packages
import os
import time
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from dataset import Dataset
from tensorflow import keras
from tensorflow.keras import backend as K
from resnet import Resnet
from metrics import compute_metrics, print_metrics, metrics_to_string, metrics_header
from losses import categorical_ce_beta_regularized, categorical_ce_poisson_regularized, categorical_ce_binomial_regularized, \
	categorical_ce_exponential_regularized
from stick_breaking import stick_breaking_layers

# Reset Keras Session keeping the same seeds
def reset_keras(seed):
	sess = keras.backend.get_session()
	keras.backend.clear_session()
	sess.close()
	sess = keras.backend.get_session()
	np.random.seed(seed)  # numpy seed
	tf.set_random_seed(seed)  # tensorflow seed
	random.seed(seed)  # random seed
	os.environ['TF_DETERMINISTIC_OPS'] = str(seed)
	os.environ['TF_CUDNN_DETERMINISM'] = str(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	config = tf.ConfigProto(allow_soft_placement=True) #, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
	config.gpu_options.allow_growth = True
	keras.backend.set_session(tf.Session(config=config))

reset_keras(1)

# Create experiment identifier
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

# Create results file header if it is the first fold
os.makedirs(args.results_path, exist_ok=True)
results_file_path = args.results_path / f'{timestamp}_{args.name}.csv'
cm_file_path = args.results_path / f'{timestamp}_{args.name}.txt'
if args.fold == 0:
	with results_file_path.open('a') as f:
		f.write(f"{metrics_header()},Time (s)")

# Load dataset
ds = Dataset(args.dataset, seed=args.seed)
ds.load(ds._name)
ds.n_folds = 10

# Parameters settings
ETA = 0.85
loss_options = {
	'ce': 'categorical_crossentropy',
	'ce_poisson': categorical_ce_poisson_regularized(ds.num_classes, eta=ETA),
	'ce_binomial': categorical_ce_binomial_regularized(ds.num_classes, eta=ETA),
	'ce_exponential': categorical_ce_exponential_regularized(ds.num_classes, eta=ETA, tau=1.0),
	'ce_beta': categorical_ce_beta_regularized(ds.num_classes, eta=ETA)
}
LOSS = loss_options[args.loss]
METRICS = ['accuracy']
BATCH_SIZE = args.batch
EPOCHS = 100


for fold in range(args.fold, ds.n_folds):
	ds.set_fold(fold)

	# Remove old graph and create a new one keeping the same seeds
	reset_keras(1)

	# Create resnet model
	l_in = keras.layers.Input(ds.sample_shape)
	output = Resnet(l_in)
	output = keras.layers.Flatten()(output)

	# Create model output
	if args.output == 'softmax':
		output = keras.layers.Dense(ds.num_classes, kernel_initializer=keras.initializers.he_normal(),
									bias_initializer=keras.initializers.Constant(0))(output)
		output = keras.layers.Softmax()(output)
	elif args.output == 'stick':
		output = stick_breaking_layers(output, ds.num_classes)

	model = keras.models.Model(l_in, output, name='Resnet')

	# Compile the model
	model.compile(
		optimizer=keras.optimizers.Adam(lr=1e-4),
		loss=LOSS,
		metrics=METRICS
	)

	# Show model info
	model.summary()
	print(F'Model compiled with {args.loss} loss and {args.output} output.')

	# Save initial time
	start_time = time.time()

	# Train model
	model.fit(
		ds.generate_train(BATCH_SIZE, {"flip_horizontal" : 1, "zoom_range" : 0.2, "width_shift_range" : 0.1}),
		epochs=EPOCHS,
		workers=7,
		use_multiprocessing=False,
		max_queue_size=BATCH_SIZE,
		validation_data=ds.generate_val(BATCH_SIZE),
		class_weight=ds.get_class_weights(),
		callbacks=[
			keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8, verbose=1, mode='min', min_lr=1e-6),
			keras.callbacks.EarlyStopping(patience=20, min_delta=0, restore_best_weights=True, verbose=1)
		],
		verbose=1
	)

	# Get test predictions
	predictions = model.predict_generator(ds.generate_test(BATCH_SIZE),
										  steps=ds.num_batches_test(BATCH_SIZE),
										  workers=7,
										  use_multiprocessing=False,
										  max_queue_size=BATCH_SIZE,
										  verbose=1)

	# Get test targets from generator
	y_set = None
	for x, y in ds.generate_test(BATCH_SIZE):
		y_set = np.array(y) if y_set is None else np.vstack((y_set, y))


	# Compute all the metrics
	metrics = compute_metrics(y_set, predictions, ds.num_classes)

	# Print metrics
	print_metrics(metrics)

	# Write test results to file
	with results_file_path.open('a') as f:
		f.write(F"\n{metrics_to_string(metrics)},{time.time() - start_time}")

	# Write confussion matrix to text file
	with cm_file_path.open('a') as f:
		f.write(f"===== Fold {fold} ======\n\n")
		f.write(str(metrics['Confusion matrix']))
		f.write("\n\n\n\n")

	del model

