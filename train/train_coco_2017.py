import os, sys, argparse, logging, torch, random, cv2

from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
	CityscapesEvaluator,
	COCOEvaluator,
	COCOPanopticEvaluator,
	DatasetEvaluators,
	LVISEvaluator,
	PascalVOCDetectionEvaluator,
	SemSegEvaluator,
	verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

DEBUG = False

# Arg parser
def argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Detectron2 Training")
    parser.add_argument(
    	"--config-file", 
    	'-cf',
    	default="", 
    	metavar="FILE", 
    	help="path to config file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
    	'--dataset-name',
    	'-dn',
    	dest='dataset_name',
    	help='Name of dataset',
    	type=str,
    	default='coco_2017'
    )
    parser.add_argument(
    	'--train-gt',
    	'-tgt',
    	dest='train_gt',
    	help='Path to train json',
    	type=str,
    	default=None
    )
    parser.add_argument(
    	'--val-gt',
    	'-vgt',
    	dest='val_gt',
    	help='Path to train json',
    	type=str,
    	default=None
    )
    parser.add_argument(
    	'--train-dir',
    	'-tdir',
    	dest='train_dir',
    	help='Path to train directory',
    	type=str,
    	default=None
    )
    parser.add_argument(
    	'--val-dir',
    	'-vdir',
    	dest='val_dir',
    	help='Path to val directory',
    	type=str,
    	default=None
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG",
    )
    parser.add_argument(
    	'--cuda',
    	'-cu',
    	dest='cuda',
    	help='CUDA card to use',
    	type=str,
    	default='0'
    )
    parser.add_argument(
    	'--batch-size',
    	'-bs',
    	dest='batch_size',
    	help='Batch size',
    	type=int,
    	default=16
    )
    parser.add_argument(
    	'--learning-rate',
    	'-lr',
    	dest='learning_rate',
    	help='Learning Rate',
    	type=float,
    	default=0.0001
    )
    return parser.parse_args()

# Random sample check
def random_meta_check(dataset_dicts, dataset_metadata, name='Test'):
	for d in random.sample(dataset_dicts, 3):
	    img = cv2.imread(d["file_name"])
	    visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
	    vis = visualizer.draw_dataset_dict(d)
	    cv2.imshow(name, vis.get_image()[:, :, ::-1])
	    k = cv2.waitKey(0)
	    if k == 27:
	    	cv2.DestroyAllWindows()

# Config setup
def setup(args):
	"""
	Create configs and perform basic setups.
	"""
	out_dir = args.config_file.split(os.sep)[-1].rsplit('.', 1)[0]
	cfg = get_cfg()
	cfg.OUTPUT_DIR = f'./{out_dir}'
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.SOLVER.IMS_PER_BATCH = args.batch_size
	cfg.SOLVER.BASE_LR = args.learning_rate
	cfg.freeze()
	print(f'cfg.DATASETS: {cfg.DATASETS}')
	default_setup(cfg, args)
	return cfg

class Trainer(DefaultTrainer):
	"""
	We use the "DefaultTrainer" which contains pre-defined default logic for
	standard training workflow. They may not work for you, especially if you
	are working on a new research project. In that case you can use the cleaner
	"SimpleTrainer", or write your own training loop. You can use
	"tools/plain_train_net.py" as an example.
	"""

	@classmethod
	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
		"""
		Create evaluator(s) for a given dataset.
		This uses the special metadata "evaluator_type" associated with each builtin dataset.
		For your own dataset, you can simply create an evaluator manually in your
		script and do not have to worry about the hacky if-else logic here.
		"""
		if output_folder is None:
			output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
		evaluator_list = []
		evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
		if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
			evaluator_list.append(
				SemSegEvaluator(
					dataset_name,
					distributed=True,
					num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
					ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
					output_dir=output_folder,
				)
			)
		if evaluator_type in ["coco", "coco_panoptic_seg"]:
			evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
		if evaluator_type == "coco_panoptic_seg":
			evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
		elif evaluator_type == "cityscapes":
			assert (
				torch.cuda.device_count() >= comm.get_rank()
			), "CityscapesEvaluator currently do not work with multiple machines."
			return CityscapesEvaluator(dataset_name)
		elif evaluator_type == "pascal_voc":
			return PascalVOCDetectionEvaluator(dataset_name)
		elif evaluator_type == "lvis":
			return LVISEvaluator(dataset_name, cfg, True, output_folder)
		if len(evaluator_list) == 0:
			raise NotImplementedError(
				"no Evaluator for the dataset {} with the type {}".format(
					dataset_name, evaluator_type
				)
			)
		elif len(evaluator_list) == 1:
			return evaluator_list[0]
		return DatasetEvaluators(evaluator_list)

	@classmethod
	def test_with_TTA(cls, cfg, model):
		logger = logging.getLogger("detectron2.trainer")
		# In the end of training, run an evaluation with TTA
		# Only support some R-CNN models.
		logger.info("Running inference with test-time augmentation ...")
		model = GeneralizedRCNNWithTTA(cfg, model)
		evaluators = [
			cls.build_evaluator(
				cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
			)
			for name in cfg.DATASETS.TEST
		]
		res = cls.test(cfg, model, evaluators)
		res = OrderedDict({k + "_TTA": v for k, v in res.items()})
		return res

def main(args):
	# Gets and sets up config
	cfg = setup(args)

	# If eval only
	if args.eval_only:
		model = Trainer.build_model(cfg)
		DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
			cfg.MODEL.WEIGHTS, resume=args.resume
		)
		res = Trainer.test(cfg, model)
		if comm.is_main_process():
			verify_results(cfg, res)
		if cfg.TEST.AUG.ENABLED:
			res.update(Trainer.test_with_TTA(cfg, model))
		return res

	"""
	If you'd like to do anything fancier than the standard training logic,
	consider writing your own training loop or subclassing the trainer.
	"""
	trainer = Trainer(cfg)
	trainer.resume_or_load(resume=args.resume)
	if cfg.TEST.AUG.ENABLED:
	    trainer.register_hooks(
	        [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
	    )
	return trainer.train()


if __name__ == '__main__':
	# Get args
	args = argument_parser()
	print("Command Line Args:", args)

	# Set DEBUG
	DEBUG = args.debug

	# Set CUDA Card
	# Cuda device setup
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda

	# Register coco train datasets
	# register_coco_instances(
	# 	f'{args.dataset_name}_train', 
	# 	{}, 
	# 	args.train_gt, 
	# 	args.train_dir
	# )
	
	# Register coco valid
	# register_coco_instances(
	# 	f'{args.dataset_name}_val', 
	# 	{}, 
	# 	args.val_gt, 
	# 	args.val_dir
	# )

	# Test train and valid annotations
	if DEBUG:
		coco_train_metadata = MetadataCatalog.get('coco_2017_train')
		coco_train_dicts = DatasetCatalog.get('coco_2017_train')
		random_meta_check(coco_train_dicts, coco_train_metadata, 'Train')
		coco_val_metadata = MetadataCatalog.get('coco_2017_val')
		coco_val_dicts = DatasetCatalog.get('coco_2017_val')	
		random_meta_check(coco_val_dicts, coco_val_metadata, 'Valid')

	# Launches main
	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)