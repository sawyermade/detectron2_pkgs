import cv2, os, sys, numpy as np, glob, imageio

def find_center(grs):
	"""
	Compute mean center of all GraspRectangles
	:return: float, mean centre of all GraspRectangles
	"""
	return np.mean(np.vstack(grs), axis=0).astype(np.int)

def offset(points, offset):
	"""
	Offset grasp rectangle
	:param offset: array [y, x] distance to offset
	"""
	points += np.array(offset).reshape((1, 2))

def _gr_text_to_no(l, offset=(0, 0)):
	"""
	Transform a single point from a Cornell file line to a pair of ints.
	:param l: Line from Cornell grasp file (str)
	:param offset: Offset to apply to point positions
	:return: Point [y, x]
	"""
	x, y = l.split()
	return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]

def load_from_cornell_file(fname):
	"""
	Load grasp rectangles from a Cornell dataset grasp file.
	:param fname: Path to text file.
	:return: GraspRectangles()
	"""
	grs = []
	with open(fname) as f:
		while True:
			# Load 4 lines at a time, corners of bounding box.
			p0 = f.readline()
			if not p0:
				break  # EOF
			p1, p2, p3 = f.readline(), f.readline(), f.readline()
			try:
				gr = np.array([
					_gr_text_to_no(p0),
					_gr_text_to_no(p1),
					_gr_text_to_no(p2),
					_gr_text_to_no(p3)
				])

				grs.append(gr)

			except ValueError:
				# Some files contain weird values.
				continue
	return grs

def find_files(cornell_dir):
	# Finds all gt files
	graspf_gt = glob.glob(os.path.join(cornell_dir, '*', 'pcd*cpos.txt'))
	graspf_gt.sort()

	# Gets all depth and rgb paths
	depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf_gt]
	rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]

	return graspf_gt, depthf, rgbf

def main():
	# Gets path to cornell directory
	cornell_dir = sys.argv[1]
	output_size = 300
	output_dir = './outputs'

	# Creates output dir if doesnt exist
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Gets all file paths
	graspf_gt, depthf, rgbf = find_files(cornell_dir)

	# Goes through all gt files
	for i in range(len(graspf_gt)):
		# Gets positive gt grasping recs for image
		grs = load_from_cornell_file(graspf_gt[i])

		# Finds center
		center = find_center(grs)

		# Gets left and top
		top_left = (
			max(0, min(center[1] - output_size // 2, 640 - output_size)), 
			max(0, min(center[0] - output_size // 2, 480 - output_size))
		)

		# Gets bottom_right
		bottom_right = (
			min(480, top_left[0] + output_size), 
			min(640, top_left[1] + output_size)
		)

		# Opens rgb image and crops it
		rgb_img = imageio.imread(rgbf[i])
		rgb_crop = rgb_img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

		# Writes images
		fname = rgbf[i].split(os.sep)[-1]
		imageio.imwrite(os.path.join(output_dir, fname), rgb_crop)
		if i % 100 == 0:
			print(fname)

if __name__ == '__main__':
	main()