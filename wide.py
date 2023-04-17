import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

t_s = 1.47583e-05
t_w = 6.469449853897094e-10
t_f = 1.8141993761502122e-09

def model(r,m,n,p,q):
    return r * (8 * (((m * t_w) / p) + ((n * t_w) / q) + (2 * t_w) + t_s) + ((m * n * t_f)/(p * q)))

def wideModel(r,w,m,n,p,q):
    return math.floor(r / w) * (
        8 * (((m * t_w) / p) + ((n * t_w) / q) + (2 * t_w) + t_s)
        + sum([((m / p) + (2 * w) - (2 * i)) * ((n / q) + (2 * w) - (2 * i)) * t_f for i in range(1, w + 1)])
    )

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
duration_o = [0] * 5
duration_2 = [
    1.21e+00,
    3.65e-01,
    2.83e-01,
    1.85e-01,
    7.15e-02,
    3.83e-02
]
duration_w_2 = [0] * 6

d_type = (["Actual"] * 5) + (["Predicted"] * 5) + (["Actual (Wide)"] * 5) + (["Predicated (Wide)"] * 5)
d_type_2 = (["Actual"] * 6) + (["Predicted"] * 6) + (["Actual (Wide)"] * 6) + (["Predicated (Wide)"] * 6)

duration_pred = [model(100,2048,2048,p,q) for (p,q) in [(1,48),(2,24),(3,16),(4,12),(6,8)]]
duration_w_pred = [wideModel(100,2048,2048,p,q) for (p,q) in [(1,48),(2,24),(3,16),(4,12),(6,8)]]
duration_2_pred = [model(100,2048,2048,p,p) for p in [1,2,3,4,5,6]]
duration_w_2_pred = [wideModel(100,2048,2048,p,p) for p in [1,2,3,4,5,6]]

def pch(initial, final):
    return 100.0 * ((final - initial) / abs(initial))

dp_pch_a = [pch(duration_pred[i], duration_w_pred[i]) for i in range(5)]
dp_pch = sum(dp_pch_a) / len(dp_pch_a)
dp_pch_2_a = [pch(duration_2_pred[i], duration_w_2_pred[i]) for i in range(5)]
dp_pch_2 = sum(dp_pch_2_a) / len(dp_pch_2_a)
print("PCH:", dp_pch, "PCH 2:", dp_pch_2, "AVG:", (dp_pch + dp_pch_2) / 2.0)

#duration_pred = [model(100,3000,3000,p,q) for (p,q) in [(1,192),(2,96),(3,64),(4,48),(6,32),(8,24),(12,16)]]
#duration_2_pred = [model(100,3000,3000,p,p) for p in [1,2,4,8,13]]

# d_type = (["Actual"] * 7)# + (["Predicted"] * 5)
# d_type_2 = (["Actual"] * 5)# + (["Predicted"] * 5)
# d_type = (["x1"] * 7) + (["x10"] * 7)
# d_type_2 = (["x1"] * 5) + (["x10"] * 5)

# t_w *= 10.0
# duration_pred = [model(100,2048,2048,p,q) for (p,q) in [(1,48),(2,24),(3,16),(4,12),(6,8)]]
# duration_w_pred = [wideModel(100,2048,2048,p,q) for (p,q) in [(1,48),(2,24),(3,16),(4,12),(6,8)]]
# duration_2_pred = [model(100,2048,2048,p,p) for p in [1,2,3,4,5,6]]
# duration_w_2_pred = [wideModel(100,2048,2048,p,p) for p in [1,2,3,4,5,6]]

# duration_pred_10 = [model(100,3000,3000,p,q) for (p,q) in [(1,192),(2,96),(3,64),(4,48),(6,32),(8,24),(12,16)]]
# duration_2_pred_10 = [model(100,3000,3000,p,p) for p in [1,2,4,8,13]]

print("Rectangular:", list(map(lambda a: "{:e}".format(a), duration_pred)))
print("Square:", list(map(lambda a: "{:e}".format(a), duration_2_pred)))
print("Rectangular Wide:", list(map(lambda a: "{:e}".format(a), duration_w_pred)))
print("Square Wide:", list(map(lambda a: "{:e}".format(a), duration_w_2_pred)))
# print("Rectangular x10:", list(map(lambda a: "{:e}".format(a), duration_pred_10)))
# print("Square x10:", list(map(lambda a: "{:e}".format(a), duration_2_pred_10)))

data = pd.DataFrame({
    "Aspect Ratio": processes + processes + processes + processes,
    "Duration (S)": duration + duration_pred + duration_o + duration_w_pred,
    "Type": d_type
})
data2 = pd.DataFrame({
    "Aspect Ratio": processes_2 + processes_2 + processes_2 + processes_2,
    "Duration (S)": duration_2 + duration_2_pred + duration_w_2 + duration_w_2_pred,
    "Type": d_type_2
})

#data_10 = pd.DataFrame({"Aspect Ratio": processes + processes, "Duration (S)": duration_pred + duration_pred_10, "t_w multiplier": d_type})
#data2_10 = pd.DataFrame({"Aspect Ratio": processes_2 + processes_2, "Duration (S)": duration_2_pred + duration_2_pred_10, "t_w multiplier": d_type_2})

# f, axes = plt.subplots(1, 2, sharey=True)
#sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data, hue = "Type", palette=sns.color_palette("deep"))
#plt.ylim(0.010, 0.04)
plt.legend(prop={"size": 9})
sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data2, hue = "Type", palette=sns.color_palette("deep"))

#sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data_10, hue = "t_w multiplier")
# sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data2_10, hue = "t_w multiplier")

# plt.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))
plt.title("2D Non-Wide vs Wide Square\nProcess Grid Aspect Ratio Advection Time")
#plt.title("Predicted Rectangular Process Grid Aspect Ratio Advection Time")
plt.show()
