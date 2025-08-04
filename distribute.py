import matplotlib.pyplot as plt
import numpy as np

# Define the number of bits (neurons)
num_bits = 4  

# Define weights for different encoding schemes
binary_weights = np.array([1, 2, 4, 8])  # Standard binary encoding
freq_weights = np.array([1, 1, 1, 1])  # Frequency encoding (equal weights)
intermediate_weights = np.array([1, 1.5, 3, 5])  # Intermediate encoding scheme

# Generate all possible encoding combinations (0000 - 1111)
num_combinations = 2 ** num_bits
all_combinations = np.array([[int(b) for b in format(i, f'0{num_bits}b')] for i in range(num_combinations)])

# Compute encoded values
binary_values = np.dot(all_combinations, binary_weights)
freq_values = np.dot(all_combinations, freq_weights)
intermediate_values = np.dot(all_combinations, intermediate_weights)

# Compute unique encoded values and their probabilities
unique_binary, counts_binary = np.unique(binary_values, return_counts=True)
unique_freq, counts_freq = np.unique(freq_values, return_counts=True)
unique_intermediate, counts_intermediate = np.unique(intermediate_values, return_counts=True)

# Normalize probabilities
counts_binary = counts_binary / counts_binary.sum()
counts_freq = counts_freq / counts_freq.sum()
counts_intermediate = counts_intermediate / counts_intermediate.sum()

# Plot distributions with better separation
plt.figure(figsize=(12, 6))
bar_width = 0.3

plt.bar(unique_binary - bar_width, counts_binary, width=bar_width, alpha=0.7, label="Standard Binary Encoding", color='b', edgecolor='black')
plt.bar(unique_freq, counts_freq, width=bar_width, alpha=0.7, label="Frequency Encoding", color='g', edgecolor='black')
plt.bar(unique_intermediate + bar_width, counts_intermediate, width=bar_width, alpha=0.7, label="Non-Fixed Weighting", color='r', edgecolor='black')

# Labels and visualization enhancements
plt.xlabel("Encoded Value")
plt.ylabel("Probability")
plt.title("Comparison of Encoding Distributions")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, max(unique_binary) + 2, 1))  # Ensure discrete tick marks
plt.savefig("distribute.png")
