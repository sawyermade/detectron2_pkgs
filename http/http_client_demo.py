import os, sys, requests, pyrealsense2 as rs
import numpy as np, cv2, jsonpickle, argparse

# Arg parser
def get_parser():
	parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
	parser.add_argument(
		'--ip',
		'-ip',
		dest='ip',
		help='Server IP',
		default='127.0.0.1',
		type=str
	)
	parser.add_argument(
		'--port',
		'-p',
		dest='port',
		help='Server Port',
		default='665',
		type=str
	)
	parser.add_argument(
		'--width',
		'-wt',
		dest='width',
		help='Image Capture Width',
		default=640,
		type=int
	)
	parser.add_argument(
		'--height',
		'-ht',
		dest='height',
		help='Image Capture height',
		default=480,
		type=int
	)
	parser.add_argument(
		"--webcam", 
		'-wc',
		dest='webcam',
		action="store_true", 
		help="Take inputs from webcam."
	)
	parser.add_argument(
		"--vis-only", 
		'-vo',
		dest='vis_only',
		action="store_true", 
		help="Visualize Return Only"
	)
	parser.add_argument(
		'--cam-num',
		'-cn',
		dest='cam_num',
		help='Webcam Number if not 0',
		default=0,
		type=int
	)
	return parser.parse_args()

# Uploads to Detectron
def upload(url, frame, vis_only=True):
	# Prep headers for http req
	content_type = 'application/json'
	headers = {'content_type': content_type}

	# jsonpickle the numpy frame
	_, frame_png = cv2.imencode('.png', frame)
	upload_list = [frame_png, vis_only]
	frame_json = jsonpickle.encode(upload_list)

	# Post and get response
	try:
		response = requests.post(url, data=frame_json, headers=headers)
		if response.text:
			# Decode response and return it
			retList = jsonpickle.decode(response.text)
			retList[0] = cv2.imdecode(retList[0], cv2.IMREAD_COLOR)

			if not vis_only:
				retList[-1] = [cv2.imdecode(m, cv2.IMREAD_GRAYSCALE) for m in retList[-1]]
			
			# returns [vis.png, bbList, labelList, scoreList, maskList]
			return retList
		else:
			return None
	except:
		return None

def main():
	# Arguments
	args = get_parser()
	url = f'http://{args.ip}:{args.port}'

	# Starts captures if not webcam
	if args.webcam:
		cap = cv2.VideoCapture(args.cam_num)
	else :
		width, height = args.width, args.height
		pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
		config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
		profile = pipeline.start(config)

	print('Hold Ctrl+C to stop...')
	while True:
		# Get frames
		if args.webcam:
			ret, frame = cap.read()
		else:
			frames = pipeline.wait_for_frames()
			frame = np.asanyarray(frames.get_color_frame().get_data())
		
		# Sends to detectron
		# returns [vis.png, bbList, labelList, scoreList, maskList]
		retList = upload(url, frame, args.vis_only)
		if not retList:
			continue

		# Shows img
		visImg = retList[0]
		visImg = cv2.resize(visImg, (1200, 900))
		cv2.imshow('Inference', visImg)
		k = cv2.waitKey(1)
		# k = cv2.waitKey(1)
		# if k == 27:
		# 	cv2.destroyAllWindows()
		# 	break

if __name__ == '__main__':
	main()