import matplotlib.pyplot as plt

in_file = open("data", "r")
line = in_file.read()
string_list = line.split()
value_list = []
for s in string_list:
    value_list.append(float(s))
in_file.close()
x = range(1,10001)

plt.scatter(x, value_list, color='blue')
plt.show()
