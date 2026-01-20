# Generalization Experiment Results

Train on DistShift1, evaluate on DistShift1 (in-distribution) and DistShift2 (out-of-distribution).

| Algorithm | DistShift1 (train) | DistShift2 (test) | Generalization Gap |
|-----------|--------------------|--------------------|--------------------|
| PPO | 0.75 ± 0.38 | 0.00 | 0.75 |
| A2C | 0.75 ± 0.38 | 0.00 | 0.75 |
| DQN | 0.00 ± 0.00 | 0.00 | 0.00 |
