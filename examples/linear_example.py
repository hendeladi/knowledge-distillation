import numpy as np
import matplotlib.pyplot as plt
import math
import random


# library
import numpy as np
import matplotlib.pyplot as plt
n=1000
# Create data
x = np.linspace(0, 1, n)
y1 = np.sqrt(0.1-x[:int(0.316*n)]**2)
y1 = np.append(y1, np.zeros(n - int(0.316*n)))

y2 = -4 * ((x-0.45)**3) + 0.2
y2 = y2[:int(0.818*n)]
y2 = np.append(y2, np.zeros(n-int(0.818*n)))


y3 = 1.3 - x[int(0.3*n):]
y3 = np.append(np.ones(int(0.3*n)), y3)

y4 = np.ones(n)


# Basic stacked area chart.
plt.figure()
plt.fill_between(x, y1, color='b', alpha=0.5)
plt.fill_between(x, y1, y2, color='r', alpha=0.6)
plt.fill_between(x, y2, y3, color='b', alpha=0.5)
plt.fill_between(x, y3, y4, color='r', alpha=0.6)
plt.xlabel(r"$X_1$")
plt.ylabel(r"$X_2$")
plt.text(0.01, 0.2, r'$x_1^2 +x_2^2 = 0.1$', color='k', fontweight='bold')
plt.text(0.32, 0.24, r'$x_2 = -4(x_1 - 0.5)^3 +0.2$', color='k', fontweight='bold')
plt.text(0.55, 0.78, r'$x_2 = 1.3-x_1$', color='k', fontweight='bold')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend(['0', '1'])
#plt.stackplot(x,y1, y2, y3,y4,labels=['A','B','C','D'])
#plt.legend(loc='upper left')
plt.show()


'''def sigmoid(x):
    return 1/(1+np.e**(-10*x))


def sigmoid_dot(x):
    return 10*sigmoid(x) * (1 - sigmoid(x))

def g(x, y):
    if y < np.sqrt(0.1 - x**2):
        return 0
    elif y < -4 * ((x-0.5)**3) + 0.2:
        return 1
    elif y < 1.3-x:
        return 0
    else:
        return 1

# x2 = ax1 + b

n = 2000000
loss_arr = []
a_arr = []
b_arr = []
features = [(random.random(),random.random()) for i in range(n)] #(x,y)
labels = [g(feature[0], feature[1]) for feature in features]
lr = 1
epochs = 200
a =-1 #2*random.random()
b = 1.3#2*random.random()
for epoch in range(epochs):
    loss = (1 / n) * sum([(labels[i] - sigmoid(features[i][0] - a * features[i][1] - b)) ** 2 for i in range(n)])
    print(f"########## epoch{epoch} #############")
    print(f"loss = {loss:.4f},  a = {a:.3f}, b = {b:.3f}\n")
    grad_a = (2/n) * sum([                 (labels[i] - sigmoid(features[i][0] - a*features[i][1] -b )) * sigmoid_dot(features[i][0] - a*features[i][1] -b )*features[i][1]                  for i in range(n) ])
    grad_b = (2/n) * sum([     (labels[i] - sigmoid(features[i][0] - a*features[i][1] -b)) * sigmoid_dot(features[i][0] - a*features[i][1] -b)          for i in range(n)])
    a = a - lr * grad_a
    b = b - lr * grad_b

    loss_arr.append(loss)
    a_arr.append(a)
    b_arr.append(b)
    print(f"########## epoch{epoch} #############")
    print(f"loss = {loss:.4f},  a = {a:.3f}, b = {b:.3f}\n")




x = np.linspace(-5,5,100)
'''