import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

t_s = 1.47583e-05
t_w = 6.469449853897094e-10
t_f = 1.8141993761502122e-09

def model(r,m,n,p,q):
    return r * (2 * (((4 * m * t_w)/(p)) + t_s) + ((m * n * t_f)/(p * q)))

processes = [
    "1x48",
    "2x24",
    "3x16",
    "4x12",
    "6x8",
]
processes_2 = [
    "1x1",
    "2x2",
    "3x3",
    "4x4",
    "5x5",
    "6x6"
]
duration = [
    3.70e-02,
    2.73e-02,
    2.38e-02,
    3.10e-02,
    2.62e-02
]
duration_2 = [
    1.21e+00,
    3.65e-01,
    2.83e-01,
    1.85e-01,
    7.15e-02,
    3.83e-02
]

data = pd.DataFrame({"Aspect Ratio": processes, "Duration (S)": duration})
data2 = pd.DataFrame({"Aspect Ratio": processes_2, "Duration (S)": duration_2})

# f, axes = plt.subplots(1, 2, sharey=True)
# sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data)
sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data2)
# plt.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))
plt.title("Square Process Grid Aspect Ratio Advection Time")
plt.show()
