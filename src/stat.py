# Import necessary libraries
import numpy as np
from scipy import stats

# Given student scores
# student_scores = np.array([27.276, 23.973, 27.952, 23.115, 18.773, 19.943, 17.499, 22.179, 20.515])
# student_scores = np.array([43.035, 49.741, 22.777, 81.515, 38.846, 43.761, 73.168, 68.878, 20.879, 22.491])
student_scores = np.array([31.618, 44.099, 32.138, 47.713])


# Hypothesized population mean
mu = 35.7369
# mu = 22.4

# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(student_scores, mu)
print("T statistic:", t_stat)
print("P-value:", p_value)

# Setting significance level
alpha = 0.05
