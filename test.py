import numpy as np

Pk=np.load("StateTransitionProbability_leopt4.npy", allow_pickle=True)

print(Pk[1,1])