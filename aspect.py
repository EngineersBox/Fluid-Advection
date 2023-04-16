import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

t_s = 1.47583e-05
t_w = 6.469449853897094e-10
t_f = 1.8141993761502122e-09

def model(r,m,n,p,q):
    return r * (8 * (((m * t_w) / p) + ((n * t_w) / q) + (2 * t_w) + t_s) + ((m * n * t_f)/(p * q)))

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
duration_pred = [model(100,2048,2048,p,q) for (p,q) in [(1,48),(2,24),(3,16),(4,12),(6,8)]]
duration_2 = [
    1.21e+00,
    3.65e-01,
    2.83e-01,
    1.85e-01,
    7.15e-02,
    3.83e-02
]
duration_2_pred = [model(100,2048,2048,p,p) for p in [1,2,3,4,5,6]]
d_type = (["Actual"] * 5) + (["Predicted"] * 5)
d_type_2 = (["Actual"] * 6) + (["Predicted"] * 6)

print("Rectangular:", list(map(lambda a: "{:e}".format(a), duration_pred)))
print("Square:", list(map(lambda a: "{:e}".format(a), duration_2_pred)))

data = pd.DataFrame({"Aspect Ratio": processes + processes, "Duration (S)": duration + duration_pred, "Type": d_type})
data2 = pd.DataFrame({"Aspect Ratio": processes_2 + processes_2, "Duration (S)": duration_2 + duration_2_pred, "Type": d_type_2})

# f, axes = plt.subplots(1, 2, sharey=True)
sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data, hue = "Type")
#sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data2, hue = "Type")
# plt.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))
plt.title("Rectangular Process Grid Aspect Ratio Advection Time")
plt.show()
