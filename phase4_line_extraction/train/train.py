# mmdet_train.py
import argparse
from mmcv import Config
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
import os.path as osp # For cleaner path handling

# 1. Define Paths
CONFIG_FILE_PATH = 'phase4_line_extraction/models/lineformer_config.py'
WORK_DIR = './lineformer_finetune' # Directory to save checkpoints and logs

def train_model():
    # 2. Load Configuration
    cfg = Config.fromfile(CONFIG_FILE_PATH)

    # 3. Modify Config for your specific needs (e.g., logging, saving checkpoints)
    if cfg.get('work_dir', None) is None:
        # Set the work_dir to the log directory
        cfg.work_dir = WORK_DIR

    # Add custom flags or changes if necessary (e.g., if you had additional command line arguments)
    # Example: cfg.resume_from = 'path/to/checkpoint.pth'

    # 4. Build Model and Datasets
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    datasets = [build_dataset(cfg.data.train)]

    # 5. Start Training
    train_detector(
        model,
        datasets,
        cfg,
        distributed=False, # Set to True if using distributed training (multiple GPUs/nodes)
        validate=True,     # Set to True to enable validation during training
    )

if __name__ == '__main__':
    train_model()