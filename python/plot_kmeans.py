import matplotlib.pyplot as plt
from math import exp

# OPTIONS
input_model_filename = "data/kmeans_model"
input_labels_filename = "data/kmeans_labels"
input_obs_filename = "data/observations"

# read kmeans model from input file
in_file = open(input_model_filename, "r")
line = in_file.read()
v = line.split()
n_states = int(v[0])
means = []
std_devs = []
offset = 1
for i in range(0, n_states):
    means.append(float(v[offset + 2*i]))
    std_devs.append(float(v[offset + 2*i + 1]))
transitions = []
offset = 1 + n_states*2
for i in range(0, n_states):
    for j in range(0, n_states):
        transitions.append(exp(float(v[offset])))
        offset = offset + 1
initial = []
for i in range(0, n_states):
    initial.append(exp(float(v[offset])))
    offset = offset + 1
in_file.close()

# read kmeans labels from file
in_file = open(input_labels_filename, "r")
line = in_file.read()
string_list = line.split()
labels_list = []
for s in string_list:
    labels_list.append(int(s))

# read observations from input file
in_file = open(input_obs_filename, "r")
line = in_file.read()
string_list = line.split()
value_list = []
value_x = []
for i in range(0, n_states):
    value_list.append([])
    value_x.append([])
print(value_list)
input_limit = 1000000
counter = 0
for s in string_list:
    value_list[labels_list[counter]].append(float(s))
    value_x[labels_list[counter]].append(counter)
    counter = counter + 1
    if counter > input_limit:
        break
in_file.close()

for i in range(0, n_states):
    print("State",i,"- Mean:",means[i],"- StdDev:",std_devs[i])

x = range(1, counter+1)
means_plot = []
std_low_plot = []
std_high_plot = []
for i in range(0, n_states):
    means_plot.append([means[i]] * len(x))
    std_low_plot.append([means[i] - 3*std_devs[i]] * len(x))
    std_high_plot.append([means[i] + 3*std_devs[i]] * len(x))

# plt.scatter(x, value_list, color='blue')
for i in range(0, n_states):
    # -- either differentiate clusters
    plt.scatter(value_x[i], value_list[i])
    # -- or just print observations
    # plt.scatter(value_x[i], value_list[i], color='blue')

    plt.plot(x, means_plot[i], color='black')
    # plt.plot(x, std_low_plot[i], color='red')
    # plt.plot(x, std_high_plot[i], color='red')
plt.show()
