import numpy as np
from src.statistics import multinomial_coeff
import matplotlib.pyplot as plt


n_range = range(2,500)


##############################
mutli_arr = []

for n in n_range:
    if n%10 == 0:
        print(n)
    s = 0
    for ka in range(0, n + 1):
        for kb in range(0, min(ka, n-ka)+1):
            s += multinomial_coeff(n, [ka, kb, n-ka-kb])*(0.2**ka)*(0.3**kb)*(0.5**(n-ka-kb))
    mutli_arr.append(s)
mutli_arr1 = mutli_arr

mutli_arr = []

for n in n_range:
    if n%10 == 0:
        print(n)
    s = 0
    for ka in range(0, n + 1):
        for kb in range(0, min(ka, n-ka)+1):
            s += multinomial_coeff(n, [ka, kb, n-ka-kb])*(0.19**ka)*(0.31**kb)*(0.5**(n-ka-kb))
    mutli_arr.append(s)
mutli_arr2 = mutli_arr


plt.figure()
plt.plot(n_range, mutli_arr1, 'b', n_range, mutli_arr2, 'r')
plt.title("multi rates")
plt.legend(["multi1", "multi2"])
plt.xlabel("number of training examples")
plt.ylabel("probability to be delta far")



plt.figure()
plt.plot(n_range, np.array(mutli_arr1)/np.array(mutli_arr2), 'g')
plt.title("multi rate")
plt.legend(['ratio'])
plt.xlabel("number of training examples")
plt.ylabel("ratio")
plt.show()
##############################

'''delta = 0.99
exp_arr = [delta**n for n in n_range]



plt.figure()
plt.plot(n_range, mutli_arr, 'b', n_range, exp_arr, 'r')
plt.title("exp rate vs multi rate")
plt.legend(["multi", "exp"])
plt.xlabel("number of training examples")
plt.ylabel("probability to be delta far")
plt.show()'''
