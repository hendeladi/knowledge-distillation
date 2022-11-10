import numpy as np
from matplotlib import pyplot as plt
from simulations.utils import save_fig
import os

baseline_term2 = np.load(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\example2\delta_far_prob_term2_baseline_arr.npy')
kd_delta_prob = np.load(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\example2\delta_far_prob_kd_arr.npy')
baseline_delta_prob = np.load(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\example2\delta_far_prob_baseline_arr.npy')
est_delta_far_prob = kd_delta_prob + baseline_term2*(1-kd_delta_prob)
n_range = list(range(2, 85))
fig1 = plt.figure(1)
plt.plot(n_range, baseline_delta_prob, 'b', n_range, est_delta_far_prob, 'r')
plt.title("estimation Vs measured delta far prob")
plt.legend(["Pr(R > delta)", "estimated"])
plt.xlabel("number of examples")
plt.ylabel("Probability")
save_fig(fig1, r"C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\latex images\final\example2", "estimated_delta_far_prob.png")

fig2 = plt.figure(2)
plt.plot(n_range, baseline_delta_prob/est_delta_far_prob, 'b',)
plt.title("ratio between measured and estimated delta far prob")
plt.legend(["Pr(R > delta) / estimated"])
plt.xlabel("number of examples")
plt.ylabel("ratio")
plt.ylim(0.85,1.01)
save_fig(fig2, r"C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\latex images\final\example2", "estimated_delta_far_prob_ratio.png")
plt.show()
print('')





baseline_delta_prob = np.load(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\example1\delta_far_prob_baseline_arr.npy')
baseline_term1 = np.load(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\example1\delta_far_prob_term1_baseline_arr.npy')
baseline_term2 = np.load(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\example1\delta_far_prob_term2_baseline_arr.npy')
kd_delta_prob = np.load(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\example1\delta_far_prob_kd_arr.npy')
kd_term1 = np.load(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\example1\delta_far_prob_term1_kd_arr.npy')
kd_term2 = np.load(r'C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\sim_results\example1\delta_far_prob_term2_kd_arr.npy')

r'''
n_range = list(range(2, 60))
fig1 = plt.figure(1)
plt.plot(n_range, baseline_delta_prob, 'b', n_range, baseline_term1, 'r', n_range, baseline_term2, 'g')
plt.legend(["Pr(R > delta)", "term1", "term2"])
plt.xlabel("number of examples")
plt.ylabel("Probability")
plt.title("Baseline probabilities (delta=0.1)")
save_fig(fig1, r"C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\latex images\final\example1", "baseline_delta_probs.png")

fig2 = plt.figure(2)
plt.plot(n_range, kd_delta_prob, 'b', n_range, kd_term1, 'r', n_range, kd_term2, 'g')
plt.legend(["Pr(R > delta)", "term1", "term2"])
plt.xlabel("number of examples")
plt.ylabel("Probability")
plt.title("KD probabilities (delta=0.1)")
save_fig(fig2, r"C:\Users\AHENDEL\OneDrive - Qualcomm\Desktop\master thesis\latex images\final\example1", "kd_delta_probs.png")

plt.show()
print('')
'''