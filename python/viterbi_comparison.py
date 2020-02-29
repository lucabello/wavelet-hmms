in_file = open("results/wahmm_viterbi", "r")
in_line = in_file.read()
in_string_list = in_line.split()
value_list = []
for s in in_string_list:
    value_list.append(int(s))
in_file.close()

compressed_file = open("results/wahmm_viterbi_compressed", "r")
compressed_line = compressed_file.read()
compressed_string_list = compressed_line.split()
compressed_list = []
for s in compressed_string_list:
    compressed_list.append(int(s))
compressed_file.close()

difference_count = 0
for i in range(0, len(value_list)):
    if value_list[i] != compressed_list[i]:
        difference_count = difference_count + 1
        if difference_count == 1 :
            print("-- first difference at time", i) # indexing starts from 0

print("Number of matches: ", len(value_list)-difference_count)
print("Number of differences: ", difference_count)
