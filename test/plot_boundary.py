import matplotlib.pyplot as plt
import numpy as np

b0 = 0.5
b1 = 0.8

x = np.arange(start=0,stop=1,step=0.001)
y = np.zeros(len(x))
y[((x>=0.5) & (x<0.8)) | ((x>=0.9) & (x<0.95))] = 1

plt.figure()
plt.plot(x,y)
plt.xlim(0,1)
plt.ylim(-0.01,1.1)
plt.xlabel('x')
plt.ylabel('g(x)')
plt.xticks(np.arange(0,1,0.1))

plt.show()
