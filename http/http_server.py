import os, sys, cv2, numpy as np, argparse, multiprocessing as mp 
import flask, glob, time, jsonpickle, torch, time, pyrealsense2
from flask_ngrok import run_with_ngrok

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine.defaults import DefaultPredictor
# from predictor import VisualizationDemo

def get_parser():
	parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
	parser.add_argument(
		"--config-file",
		'-cf',
		default="../configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--confidence-threshold",
		'-ct',
		type=float,
		default=0.5,
		help="Minimum score for instance predictions to be shown",
	)
	parser.add_argument(
		"--opts",
		'-op',
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=[],
		nargs=argparse.REMAINDER,
	)
	parser.add_argument(
		'--cuda',
		'-cu',
		dest='cuda',
		help='Enter cuda card number to use as integer',
		default='0',
		type=str
	)
	parser.add_argument(
		'--ip',
		'-ip',
		dest='ip',
		help='Server IP',
		default='0.0.0.0',
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
		"--ngrok", 
		'-ng',
		dest='ngrok',
		action="store_true", 
		help="Uses Ngrok."
	)
	return parser.parse_args()

# Config setup
def setup_cfg(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	
	# Set score_threshold for builtin models
	cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
	cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
	cfg.freeze()
	return cfg

# Globals
args = get_parser()
DOMAIN = args.ip
PORT = args.port
app = flask.Flask(__name__)
if args.ngrok: run_with_ngrok(app)
model = None
metadata = None
cpu_device = torch.device('cpu')
instance_mode = ColorMode.IMAGE

# Cuda device setup
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

# Awaits POST requests for inference
@app.route('/', methods=['POST'])
def upload_file():
	# Get request and unzip/decode
	r = flask.request
	upload_list = jsonpickle.decode(r.data)
	vis_only = upload_list[1]
	im = cv2.imdecode(upload_list[0], cv2.IMREAD_COLOR)

	# Run inference
	predictions = model(im)
	image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	visualizer = Visualizer(image, metadata, instance_mode=instance_mode)
	vis_output = None
	pred = None

	if "panoptic_seg" in predictions:
		pred_pan, pan_info = predictions["panoptic_seg"]
		vis_output = visualizer.draw_panoptic_seg_predictions(
            pred_pan.to(cpu_device), pan_info
        )
	
	else:
		if "sem_seg" in predictions:
			pred_seg = predictions['sem_seg'].to(cpu_device)
			vis_output = visualizer.draw_sem_seg(
                predictions["sem_seg"].argmax(dim=0).to(cpu_device)
            )
		
		if "instances" in predictions:
			pred = predictions["instances"].to(cpu_device)
			vis_output = visualizer.draw_instance_predictions(predictions=pred)

	# If any detections of instances
	if pred and vis_output:
		
		# Pulls all information
		bbList = pred.pred_boxes.tensor.numpy()
		labelList = pred.pred_classes.numpy()
		maskList = pred.pred_masks.numpy()
		scoreList = pred.scores.numpy()
		class_names = metadata.get("thing_classes", None)
		labelList = [class_names[i] for i in labelList]

		# Converts visualized image to BGR
		vis_img = cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR)

		# Converts maskList
		maskOnes = np.ones(maskList[0].shape)
		maskList = [maskOnes * m for m in maskList]
		pngList = [cv2.imencode('.png', m)[1] for m in maskList]

		# Creates return list and encode masks for size
		if vis_only:
			retList = [cv2.imencode('.png', vis_img)[1]]
		else:
			retList = [cv2.imencode('.png', vis_img)[1], bbList, labelList, scoreList, pngList]

		# Encodes to jsonpickle and sends json
		retList_encoded = jsonpickle.encode(retList)

		# returns [vis.png, bbList, labelList, scoreList, maskList]
		return flask.Response(response=retList_encoded, status=200, mimetype='application/json')

	# No detections found
	else:
		return flask.Response(response=None)

def main():
	# Globals
	global model, metadata

	# Setup logger
	mp.set_start_method("spawn", force=True)
	setup_logger(name="fvcore")
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	# Setup config and predictor
	cfg = setup_cfg(args)
	# model = VisualizationDemo(cfg)
	model = DefaultPredictor(cfg)
	metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )

	# Run http app
	if args.ngrok:
		app.run()
	else:
		app.run(port=PORT, host=DOMAIN, debug=False)

if __name__ == '__main__':
	main()