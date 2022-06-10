# Reproduce Figure 9 LB
Note: 
1. job distribution has some randomness.
2. we did not set random seed since we are streaming each set of job 50 times, if each set has the exact same job distribution, we cannot calculate mean and std.
3. therefore, the output mean and std may vary a little bit, but the trend will stay the same.

## Model testing
Test UDR_1 model, output reward mean and std:
```
python rl_test.py --saved_model="results/testing_model/udr_1/model_ep_49600.ckpt"
# example output: [-4.80, 0.07]
```

Test UDR_2 model, output reward mean and std:
```
python rl_test.py --saved_model="results/testing_model/udr_2/model_ep_44000.ckpt"
# example output: [-3.87, 0.08]
```

Test UDR_3 model, output reward mean and std:
```
python rl_test.py --saved_model="results/testing_model/udr_3/model_ep_25600.ckpt"
# example output: [-3.57, 0.07]
```

Test Genet model, output reward mean and std:
```
python rl_test.py --saved_model="results/testing_model/adr/model_ep_20200.ckpt"
# example output: [-3.02, 0.04]
```

## Result plotting
Put the above output results into bar plot.
```
python analysis/fig9_lb.py
```