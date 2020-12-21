# Select GPU index that will be used
gpu=0

# Specify datasets directory
datadir=../datasets

# Specify directory where results files will be stored
resultsdir=../results/


# Retinopathy
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy stick ce -d retinopathy -n stick_ce
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy stick ce_poisson -d retinopathy -n stick_ce_poisson
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy stick ce_binomial -d retinopathy -n stick_ce_binomial
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy stick ce_exponential -d retinopathy -n stick_ce_exponential
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy stick ce_beta -d retinopathy -n stick_ce_beta

CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy softmax ce -d retinopathy -n softmax_ce
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy softmax ce_poisson -d retinopathy -n softmax_ce_poisson
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy softmax ce_binomial -d retinopathy -n softmax_ce_binomial
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy softmax ce_exponential -d retinopathy -n softmax_ce_exponential
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/retinopathy softmax ce_beta -d retinopathy -n softmax_ce_beta

# Adience
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience stick ce -d adience -n stick_ce
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience stick ce_poisson -d adience -n stick_ce_poisson
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience stick ce_binomial -d adience -n stick_ce_binomial
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience stick ce_exponential -d adience -n stick_ce_exponential
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience stick ce_beta -d adience -n stick_ce_beta

CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience softmax ce -d adience -n softmax_ce
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience softmax ce_poisson -d adience -n softmax_ce_poisson
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience softmax ce_binomial -d adience -n softmax_ce_binomial
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience softmax ce_exponential -d adience -n softmax_ce_exponential
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/adience softmax ce_beta -d adience -n softmax_ce_beta

#FGNet
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet stick ce -d fgnet -n stick_ce
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet stick ce_poisson -d fgnet -n stick_ce_poisson
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet stick ce_binomial -d fgnet -n stick_ce_binomial
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet stick ce_exponential -d fgnet -n stick_ce_exponential
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet stick ce_beta -d fgnet -n stick_ce_beta

CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet softmax ce -d adience -n softmax_ce
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet softmax ce_poisson -d fgnet -n softmax_ce_poisson
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet softmax ce_binomial -d fgnet -n softmax_ce_binomial
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet softmax ce_exponential -d fgnet -n softmax_ce_exponential
CUDA_VISIBLE_DEVICES=$gpu DATASETS_DIR=$datadir python main.py $resultsdir/fgnet softmax ce_beta -d fgnet -n softmax_ce_beta
