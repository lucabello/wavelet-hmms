import matplotlib.pyplot as plt

in_file = open("data/observations", "r")
line = in_file.read()
string_list = line.split()
value_list = []
input_limit = 10000
counter = 1
for s in string_list:
    value_list.append(float(s))
    counter = counter + 1
    if counter > input_limit:
        break
in_file.close()


path_file = open("data/path", "r")
line = path_file.read()
string_list = line.split()
path_list = []
counter = 1
for s in string_list:
    path_list.append(int(s)*10)
    counter = counter + 1
    if counter > input_limit:
        break
path_file.close()

x = range(1,input_limit+1)

plt.scatter(x, value_list, color='blue')
plt.step(x, path_list, color='red')
plt.show()
