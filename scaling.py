import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

t_s = 1.47583e-05
t_w = 6.469449853897094e-10
t_f = 1.8141993761502122e-09

def model(r,m,n,p,q):
    return r * (2 * (((4 * m * t_w)/(p)) + t_s) + ((m * n * t_f)/(p * q)))

processes = [1,2,4,8,16,32,48,1,2,4,8,16,32,48]
duration = [
    2.28e-01,
    1.16e-01,
    5.87e-02,
    3.01e-02,
    1.40e-02,
    9.42e-03,
    1.38e-02
]
duration += [model(100,2048,2048,p,1) for p in processes[7:]]
d_type = (["Actual"] * 7) + (["Predicted"] * 7)

print(processes[7:],["{:e}".format(d) for d in duration[7:]])

data = pd.DataFrame({"Processes": processes, "Duration (s)": duration, "Type": d_type})
g = sns.lineplot(x = "Processes", y = "Duration (s)", data = data, hue="Type", marker = "o")
# plt.ticklabel_format(style="scientific", axis="y", scilimits=(0,0))
plt.title("Advection Time For Varying Process Counts")
plt.show()
