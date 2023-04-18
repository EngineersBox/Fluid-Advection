import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

t_s = 1.47583e-05
t_w = 6.469449853897094e-10
t_f = 1.8141993761502122e-09

def model(r,w,m,n,p,q):
    return r * (8 * (((m * t_w) / p) + ((n * t_w) / q) + (2 * t_w) + t_s) + ((m * n * t_f)/(p * q)))

def wideModel(r,w,m,n,p,q):
    return math.floor(r / w) * (
        8 * (((m * t_w) / p) + ((n * t_w) / q) + (2 * t_w) + t_s)
        + (t_f * w * (3 * m * (n + q * (w - 1)) + p * (w - 1) * (3 * n + 4 * q * w - 2 * q)))/(3 * p * q)
    )

rect_aspect_ratios = [
    "1x48",
    "2x24",
    "3x16",
    "4x12",
    "6x8",
]
square_aspect_ratios = [
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

# duration_pred = [model(100,2048,2048,p,q) for (p,q) in [(1,48),(2,24),(3,16),(4,12),(6,8)]]
# duration_w_pred = [wideModel(100,2048,2048,p,q) for (p,q) in [(1,48),(2,24),(3,16),(4,12),(6,8)]]
# duration_2_pred = [model(100,2048,2048,p,p) for p in [1,2,3,4,5,6]]
# duration_w_2_pred = [wideModel(100,2048,2048,p,p) for p in [1,2,3,4,5,6]]

rect = [(1,48),(2,24),(3,16),(4,12),(6,8)]
square = [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)]

ar = rect 
ar_l = rect_aspect_ratios

def predict(list_w, aspect_ratios, model_func):
    result = []
    for w in list_w:
        for (p,q) in aspect_ratios:
            result.append(model_func(1024,w,2048,2048,p,q))
    return result

w_lim = 5
# halos = [2**i for i in range(w_lim)]
halos = []
for (p,q) in ar:
    halos.append(2 ** math.floor(math.log(
        min((2048.0 / float(p)) / 2.0, (2048.0 / float(q)) / 2.0),
        2
    )))

hw = []
for w in halos:
    hw += [w] * len(ar)

predictions = pd.DataFrame({
    "Aspect Ratio": ar_l * w_lim,
    "Duration (S)": predict(
        halos,
        ar,
        wideModel
    ),
    "Halo Width": hw
})
print(predictions)

# def pch(initial, final):
    # return 100.0 * ((final - initial) / abs(initial))

# dp_pch_a = [pch(duration_pred[i], duration_w_pred[i]) for i in range(5)]
# dp_pch = sum(dp_pch_a) / len(dp_pch_a)
# dp_pch_2_a = [pch(duration_2_pred[i], duration_w_2_pred[i]) for i in range(5)]
# dp_pch_2 = sum(dp_pch_2_a) / len(dp_pch_2_a)
# print("PCH:", dp_pch, "PCH 2:", dp_pch_2, "AVG:", (dp_pch + dp_pch_2) / 2.0)

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

# print("Rectangular:", list(map(lambda a: "{:e}".format(a), duration_pred)))
# print("Square:", list(map(lambda a: "{:e}".format(a), duration_2_pred)))
# print("Rectangular Wide:", list(map(lambda a: "{:e}".format(a), duration_w_pred)))
# print("Square Wide:", list(map(lambda a: "{:e}".format(a), duration_w_2_pred)))
# print("Rectangular x10:", list(map(lambda a: "{:e}".format(a), duration_pred_10)))
# print("Square x10:", list(map(lambda a: "{:e}".format(a), duration_2_pred_10)))

# data = pd.DataFrame({
    # "Aspect Ratio": processes + processes + processes + processes,
    # "Duration (S)": duration + duration_pred + duration_o + duration_w_pred,
    # "Type": d_type
# })
# data2 = pd.DataFrame({
    # "Aspect Ratio": processes_2 + processes_2 + processes_2 + processes_2,
    # "Duration (S)": duration_2 + duration_2_pred + duration_w_2 + duration_w_2_pred,
    # "Type": d_type_2
# })

#data_10 = pd.DataFrame({"Aspect Ratio": processes + processes, "Duration (S)": duration_pred + duration_pred_10, "t_w multiplier": d_type})
#data2_10 = pd.DataFrame({"Aspect Ratio": processes_2 + processes_2, "Duration (S)": duration_2_pred + duration_2_pred_10, "t_w multiplier": d_type_2})

# f, axes = plt.subplots(1, 2, sharey=True)
#sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data, hue = "Type", palette=sns.color_palette("deep"))
#plt.ylim(0.010, 0.04)
plt.legend(prop={"size": 9})
#sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data2, hue = "Type", palette=sns.color_palette("deep"))

#sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data_10, hue = "t_w multiplier")
# sns.barplot(y = "Duration (S)", x = "Aspect Ratio", data = data2_10, hue = "t_w multiplier")

sns.lineplot(y = "Duration (S)", x = "Halo Width", data = predictions, hue = "Aspect Ratio", marker = "o")

# plt.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))
plt.title("Wide Halo Rectangular\nProcess Grid Aspect Ratio Advection Time")
#plt.title("Predicted Rectangular Process Grid Aspect Ratio Advection Time")
plt.show()
