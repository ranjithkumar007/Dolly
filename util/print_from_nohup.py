import matplotlib.pyplot as plt 
import sys
import re

file_name = sys.argv[1]

with open(file_name, 'r') as f:
	lines = f.readlines()
	epochs = []
	rewards = []
	for line in lines:
		if line.startswith("On training set :"):
			line = line[18:]
			parts = line.split(' | ')
			epochs.append(int(re.sub('[a-zA-Z_:]', '', parts[0])))
			rewards.append(float(re.sub('[a-zA-Z_:]', '', parts[1])))


	plt.plot(epochs, rewards)
	plt.show()
	plt.savefig('figures/reward.png')