from glob import glob
import csv

results = glob("results*")
# output = tuple()
csv_data = list()
for res in results:
    with open(res,'r') as data:
        output = tuple(data.read().strip().split('\t'))
        csv_data.append(output)

print(csv_data)

with open('final_results.csv', 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['vec_size', 'cpu_time', 'gpu_time'])
    for row in csv_data:
        csv_out.writerow(row)