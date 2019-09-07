import numpy as np
import copy
import matplotlib.pyplot as plt
import math


""" 
    Program 6.4 Plotting program for one-body problem
    Inputs: time interval (time_interval), initial conditions ic
    if = [x0 vx0 y0 vy0], (x position, x velocity, y position, y velocity)

"""
def orbit(time_interval = np.array([0, 100]), initial_conditions = np.array([0, 1, 2, 0]), nr_steps = 10000, steps_per_point_plotted = 5):
    h = (time_interval[1] - time_interval[0]) / nr_steps
    y = np.array([initial_conditions])
    print(y.shape)
    t = np.array([time_interval[0]])

    for k in range(math.floor(nr_steps / steps_per_point_plotted)):
        for i in range(steps_per_point_plotted):
            t = np.insert(t, i+1, t[i] + h)
            print("Y of", i, " is ", y[i])
            print("insert output", np.insert(y, i+1, euler_step(t[i], y[i], h)))

    plt.plot(y[0,:, 0],y[0, :, 2], marker ='o',linestyle = '-', color = 'y')
    plt.show()



def euler_step(t, x, h):
    # One step of the Euler Method
    print(x.shape, x)
    print(yDot(t, x)),
    print(h * yDot(t,x))
    print(x + h * yDot(t,x))
    return x + h * yDot(t, x)


def yDot(t,x):
    dist = math.sqrt(x[0]**2 + x[2]**2)
    return np.array([
        x[1], 
        (-3*x[0]) / dist**3,
        x[2],
        (-3*-x[2]) / dist**3
    ])


if __name__ == "__main__":
    orbit()