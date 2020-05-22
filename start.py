import sqlite3
import diffprivlib as dpl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

comp = []
def dp_test(data, dp):
    return abs(dp.randomise(data) - data)

def dp_seq_test(data, dp, eps, sens):
    dp.set_epsilon(eps)
    dp.set_sensitivity(sens)
    #print(float(eps), int(sens))
    ds = [dp_test(data, dp) for _ in range(100)]
    #print(ds)
    #print(sum(ds))
    comp.append((eps, sens, ds))

conn = sqlite3.connect("diabetes")

conn.row_factory = lambda cursor, row: row[0]
cur = conn.cursor()

data = cur.execute("select sum(num_medications) from data").fetchone()
dp = dpl.mechanisms.Laplace()
eps = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
sens = [250, 200, 150, 100, 75, 50, 35, 20, 10]
for e in eps:
    for s in sens:
        dp_seq_test(data, dp, e, s)

fig = plt.figure()
ax = Axes3D(fig)

x = [x for (x, _, _) in comp for i in range(100)]
y = [y for (_, y, _) in comp for i in range(100)]
z = []
for (_, _, ds) in comp:
    z.extend(ds)

ax.scatter(x, y, z)
ax.set_xlabel("Epsilon")
ax.set_ylabel("Sensitivity")
ax.set_zlabel("True error")
ax.set_title("True error for different sensitivities and epsilons")
plt.show()

comp = []
eps = [10, 5, 1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
sens = [2**32, 2**28, 2**24, 2**20, 2**16, 2**8, 10000, 1000, 100, 75, 50, 35, 20, 10]
for e in eps:
    for s in sens:
        dp_seq_test(data, dp, e, s)

z = []
epsValue = 0.01
maxSens = 2**32
for (eps, sens, ds) in comp:
    if(eps == epsValue and sens <= maxSens):
        z.extend(ds)

x_2d = [sens for (eps, sens, _) in comp if eps == epsValue and sens <= maxSens for i in range(100)]
plt.scatter(x_2d, z)
m, b = np.polyfit(np.log(x_2d), np.log(z), 1)
plt.plot(x_2d, m*np.array(x_2d)+b)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Sensitivity")
plt.ylabel("True error")
plt.title("True error with different sensitivities given epsilon = " + str(epsValue))
plt.show()