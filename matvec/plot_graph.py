import pandas as pd   
import matplotlib.pyplot as plt 

results = pd.read_csv('final_results.csv', index_col='vec_size')
fig1 = plt.scatter(results.index, results.cpu_time, c = 'b')
fig2 = plt.scatter(results.index, results.gpu_time, c = 'g')

plt.show()
