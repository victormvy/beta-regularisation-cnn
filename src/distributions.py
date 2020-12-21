import math
import numpy as np
from scipy.integrate import simps
from scipy.special import gamma, softmax, comb
from scipy.stats import binom, poisson, expon


#### BETA DISTRIBUTION ####

# Compute the beta distribution function (a,b) for value x.
def beta(x, a, b):
	return gamma(a+b) / (gamma(a) * gamma(b)) * x ** (a-1) * (1 - x) ** (b-1)

def beta_mean(a, b):
	return a / (a + b)

def beta_var(a, b):
	return (a * b) / ((a + b + 1) * (a + b) ** 2)

def find_beta_params(desired_mean, min_var, max_var, a_range=100, b_range=100):
	for a in range(1, a_range + 1):
		for b in range(1, b_range + 1):
			mean = beta_mean(a, b)
			var = beta_var(a, b)

			if abs(desired_mean - mean) < 0.002 and var >= min_var and var <= max_var:
				return a, b, mean, var

def find_beta_params_for_classes(n):
	means = np.linspace(0, 1, n + 2)[1:-1]
	params = []

	for mean in means:
		params.append(find_beta_params(mean, 0.0005, 0.005))
	
	return params


# Get n evenly-spaced intervals in [0,1].
def get_intervals(n):
	points = np.linspace(0, 1.0, n + 1)
	intervals = []
	for i in range(0, points.size - 1):
		intervals.append((points[i], points[i+1]))

	return intervals

# Get probabilities from beta distribution (a,b) for n splits
def get_beta_probabilities(n, a, b):
	intervals = get_intervals(n)
	probs = []

	for interval in intervals:
		x = np.arange(interval[0], interval[1], 1e-6)
		y = beta(x, a, b)
		probs.append(simps(y,x))

	return probs



#### POISSON DISTRIBUTION ####

# Get probabilities for each true class based on a poisson distribution
# n is the number of classes
# returns a matrix where each row represents the true class and each column the probability for class n
def get_poisson_probabilities(n):
	probs = []

	for true_class in range(1, n+1):
		probs.append(poisson.pmf(np.arange(0, n), true_class))

	return softmax(np.array(probs), axis=1)


#### BINOMINAL DISTRIBUTION ####


def get_binominal_probabilities(n):
	params = {}
	
	params['4'] = [
		0.2,
		0.4,
		0.6,
		0.8
	]

	params['5'] = [
		0.1,
		0.3,
		0.5,
		0.7,
		0.9
	]

	params['6'] = [
		0.1,
		0.26,
		0.42,
		0.58,
		0.74,
		0.9
	]

	params['8'] = [
		0.1,
		0.21428571,
		0.32857143,
		0.44285714,
		0.55714286,
		0.67142857,
		0.78571429,
		0.9
	]


	probs = []

	for true_class in range(0, n):
		probs.append(binom.pmf(np.arange(0, n), n - 1, params[str(n)][true_class]))

	return np.array(probs)


#### EXPONENTIAL DISTRIBUTION ####


def get_exponential_probabilities(n, tau=1.0):
	probs = []

	for true_class in range(0, n):
		probs.append(-np.abs(np.arange(0, n) - true_class) / tau)

	return softmax(np.array(probs), axis=1)
