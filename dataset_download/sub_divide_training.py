import os, sys
from tqdm import tqdm

def sub_divide_dir(dir_path, max_files=5000):
	# Creates sorted file list
	file_list = os.listdir(dir_path)
	file_list = [f for f in file_list if os.path.isfile(os.path.join(dir_path, f))]
	file_list.sort(key=lambda x: int(x.rsplit('.', 1)[0]))

	# Checks if less/equal to max_files
	if len(file_list) <= max_files:
		return False

	# Sub divides so no directory has more than max_files
	out_dir = dir_path + '_sub'
	count = 0
	for i, filename in enumerate(tqdm(file_list)):
		# Create new sub directory
		if i % (max_files-1) == 0:
			out_sub = os.path.join(out_dir, str(count))
			# os.system(f'mkdir -p {out_sub}')
			count += 1

		# Gets file path and copies to sub directory
		file_path = os.path.join(dir_path, filename)
		# os.system(f'cp {file_path} {out_sub}')

	return True

def main():
	# Gets paths
	dataset_path = sys.argv[1]
	train_path = os.path.join(dataset_path, 'train2017')
	test_path = os.path.join(dataset_path, 'test2017')
	val_path = os.path.join(dataset_path, 'val2017')

	# Checks for second max files arg
	if len(sys.argv) > 2:
		max_files = int(sys.argv[2])
	else:
		max_files = 5000

	# Run on train, test, val 2017s
	sub_divide_dir(train_path, max_files)
	sub_divide_dir(test_path, max_files)
	sub_divide_dir(val_path, max_files)

if __name__ == '__main__':
	main()