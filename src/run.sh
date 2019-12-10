for i in {0..100}; do
    python batch_runner.py --config al.config_rl_ac --job_id $i --save_path ../log/data/;
done


# Runs

# Normal
# kappa = 0.985,0.99
# lr = 2e-4
# in_dim = 2
# sample_cost = 1

# Uniform
# kappa = 0.08,0.1
# lr = 1e-3
# in_dim = 100
# sample_cost = 1.5