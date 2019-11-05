import matplotlib.pyplot as plt
import numpy as np

TLEED_RFACTOR = 0.2794

rfactor_progress = np.loadtxt("rfactor_progress.txt")
nepochs = len(rfactor_progress)

plt.xlabel("Epoch")
plt.ylabel("R-Factor")
plt.hlines(TLEED_RFACTOR, 0, nepochs, label="TensErLEED Search")
plt.plot(np.arange(nepochs), rfactor_progress, "o-", label="Bayesian Optimization")
plt.legend()
plt.show()
