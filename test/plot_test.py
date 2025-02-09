import os
import numpy as np
import matplotlib.pyplot as plt


results_path = r'C:\Users\AHENDEL\Downloads\res'
x_axis = np.load(os.path.join(results_path, 'n_range.npy'))

Ps = np.load(os.path.join(results_path, 'delta_far_prob_realizable_arr.npy'))
cond_prob = np.load(os.path.join(results_path, 'main_term_baseline_arr.npy'))

baseline_risk = np.load(os.path.join(results_path, 'delta_far_prob_baseline_arr.npy'))
estimate = Ps + (1-Ps)*cond_prob
fig = plt.figure()
plt.plot(x_axis, baseline_risk, x_axis, estimate)

#plt.ylim([0, 0.5])
#plt.xlim([n_range[0] - 1, n_range[-1] + 1])
plt.xlabel('n')
plt.ylabel('probability')
plt.legend(['real', 'estimate'])
plt.title('Upper and lower bounds on excess error exponent')
#dest_dir = r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\ISIT\example_1\final'
#save_fig(fig, dest_dir, "error_exponent.png")
plt.show()