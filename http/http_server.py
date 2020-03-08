import os, sys, cv2, numpy as np, argparse, multiprocessing as mp, time, pyrealsense2
import flask, glob, time, jsonpickle, torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from detectron2.data import MetadataCatalog

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

def get_parser():
	parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
	parser.add_argument(
		"--config-file",
		default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--confidence-threshold",
		type=float,
		default=0.5,
		help="Minimum score for instance predictions to be shown",
	)
	parser.add_argument(
		"--opts",
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=[],
		nargs=argparse.REMAINDER,
	)
	parser.add_argument(
		'--cuda',
		dest='cuda',
		help='Enter cuda card number to use as integer',
		default='0',
		type=str
	)
	parser.add_argument(
		'--ip',
		dest='ip',
		help='Server IP',
		default='127.0.0.1',
		type=str
	)
	parser.add_argument(
		'--port',
		dest='port',
		help='Server Port',
		default='665',
		type=str
	)

	return parser.parse_args()

# Globals
args = get_parser()
DOMAIN = args.ip
PORT = args.port
app = flask.Flask(__name__)
model = None
metadata = None

# Cuda device setup
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

# Awaits POST requests for inference
@app.route('/', methods=['POST'])
def upload_file():
	# Get request and unzip/decode
	r = flask.request
	im = jsonpickle.decode(r.data)
	im = cv2.imdecode(im, cv2.IMREAD_COLOR)

	# Run inference
	# timers = defaultdict(Timer)
	# with c2_utils.NamedCudaScope(0):
	# 	cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
	# 		model, im, None, timers=timers
	# 	)
	predictions, visualized_output = model.run_on_image(im)
	# return flask.Response(response=None)

	if "panoptic_seg" in predictions:
		pred, segments_info = predictions["panoptic_seg"].to(torch.device('cpu'))
	
	else:
		if "sem_seg" in predictions:
			pred = predictions['sem_seg'].to(torch.device('cpu'))
		
		if "instances" in predictions:
			pred = predictions["instances"].to(torch.device('cpu'))

	# print(pred)
	bbList = pred.pred_boxes.tensor.numpy()
	labelList = pred.pred_classes.numpy()
	maskList = pred.pred_masks.numpy()
	scoreList = pred.scores.numpy()
	class_names = metadata.get("thing_classes", None)
	labelList = [class_names[i] for i in labelList]
	# print(f'label list:\b{labelList}')

	# # Encodes to png files
	# pngList = [cv2.imencode('.png', m)[1] for m in maskList]
	retList = [cv2.imencode('.png', visualized_output.get_image())[1], bbList, labelList, scoreList, maskList]

	# # Encodes to jsonpickle and sends json
	retList_encoded = jsonpickle.encode(retList)

	# returns [vis.png, bbList, labelList, scoreList, maskList]
	return flask.Response(response=retList_encoded, status=200, mimetype='application/json')


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
	model = VisualizationDemo(cfg)
	metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"

    )

	# Run http app
	app.run(port=PORT, host=DOMAIN, debug=False)

if __name__ == '__main__':
	main()