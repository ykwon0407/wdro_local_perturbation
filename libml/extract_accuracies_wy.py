from glob import glob
import sys, os

experiment_directory = './experiments'
new_directory ='./experiments/acc_all'
if not os.path.exists(new_directory):
	os.makedirs(new_directory)

for from_directory in sorted(glob(experiment_directory+'/l2*/*/*')):
	to_directory = new_directory+'{}'.format(from_directory[len(experiment_directory):])
	if not os.path.exists(to_directory):
		os.makedirs(to_directory)
		os.system('cp {} {}'.format(from_directory+'/accuracies.txt', to_directory+'/accuracies.txt'))
		os.system('cp {} {}'.format(from_directory+'/lipschitz.txt', to_directory+'/lipschitz.txt'))
