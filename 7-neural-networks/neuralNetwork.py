import numpy as np
import math
from scipy.stats import logistic
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D # 3D plotting

# hidden layer size
hl_size = 100
input_size = 3
data = np.loadtxt("data2Class_adjusted.txt")
inputs = data[:, :input_size]
outputs = data[:, input_size]
parameter_ini_1 = math.sqrt(3/input_size)
parameter_ini_2 = math.sqrt(3/hl_size)
w1 = np.random.random((hl_size,input_size))
w2 = np.random.random((1, hl_size))
w = [w1, w2]
alpha = 0.05

def forward(x, w):
    """
    execute for each input line
    :param x: the inputs for each line
    :param w:
    :return:
        x_new:  updated input for each layer
        z:      the result in the end, and the new x
    """
    xi = x
    x_layers = [x]
    for wi in w:
        #print "xi shape: " + str(xi.shape)
        #print "wi shape: " + str(wi.shape)
        z = np.dot(wi, xi)
        xi = logistic.cdf(z)
        x_layers.append(xi)
    return x_layers, z


def backward(delta_L_plus_1, x, w):
    """
    :param delta_L_plus_1:
    :param x:   initial inputs from file
    :param w:
    :return:
    """
    xi, f = forward(x, w)
    delta_p1 = delta_L_plus_1
    dw_list =[]
    for xb, wb in zip(reversed(xi[:-1]), reversed(w)):
        delta = np.multiply(np.dot(delta_p1, wb), np.multiply(xb, 1-xb).T)
        dw = np.dot(delta, xb.T)
        delta_p1 = delta
        dw_list.insert(0, dw)
    return dw_list


def hinge_loss(f, y):
    return max(1-f*y, 0)


def delta_l_plus_1(y, f):
    if (1-f*y)>0:
        return -1*y
    else:
        return 0

mean_loss_list = []
for i in range(0, 15):
    dw_lists = []
    hinge_loss_values = []
    for input, output in zip(inputs, outputs):
        x_list, f = forward(input, w)
        del_p1 = delta_l_plus_1(output, f)
        dw_list = backward(del_p1, input, w)
        dw_lists.append(dw_list)
        hinge_loss_values.append(hinge_loss(output, f))
    meanLoss = np.mean(hinge_loss_values)
    mean_loss_list.append(meanLoss)
    print meanLoss
    dw_values = dw_lists[0]
    for i in range(1, len(dw_lists)):
        dw_values += dw_lists[i]

    w = tuple(wi - alpha * dwi for wi, dwi in zip(w, dw_values))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mean_loss_list)
ax.set_xlabel('iter no.')
ax.set_ylabel('mean loss')

fig_3D = plt.figure()
ax = fig_3D.add_subplot(111, projection='3d')
f_list = np.array([forward(xx, w)[1] for xx in inputs])
print f_list.shape
ax.scatter(inputs[:,1],inputs[:,2],f_list,color="red")

plt.show()