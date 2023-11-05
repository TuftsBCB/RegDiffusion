You can find some scripts we used to carry out experiments for our recomb submission. 

- `recomb_benchmark.py` and `recomb_benchmark.sh` are mostly used for getting results on GENIE3/GRNBoost2. Since these two methods took a rather long time to compute, we execute experiment for each dataset separately in parallel. 

- `recomb_benchmark_all.py` and `recomb_benchmark_gpu.sh` are mostly used for gathering results for DeepSEM, DAZZLE, and RegDiffusion. 

- `recomb_benchmark_reduced.py` and `recomb_benchmark_reduced.sh` tests the impact of reduced sample size on the training effect. 

- Most execution scripts/history are stored in recomb_experiments.ipynb.

