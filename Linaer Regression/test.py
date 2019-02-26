
import numpy as np
a = np.array([1,2,3])
b = np.array([2,5,8])
x = np.vstack((a,b))
y = np.cov(a,b)
y = np.corrcoef(a, b)
pass