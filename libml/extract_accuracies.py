from glob import glob
import sys, os

experiment_directory = './data/dro-io_history'
new_directory ='../experiments/acc_all'
if not os.path.exists(new_directory):
	os.makedirs(new_directory)

for from_directory in sorted(glob(experiment_directory+'/*/*/*')):
	print(from_directory)
	to_directory = new_directory+'{}'.format(from_directory[len(experiment_directory):])
	if not os.path.exists(to_directory):
		os.makedirs(to_directory)
		os.system('cp {} {}'.format(from_directory+'/accuracies.txt', to_directory+'/accuracies.txt'))
