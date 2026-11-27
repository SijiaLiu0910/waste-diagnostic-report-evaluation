"""Train the YOLO detector on the complete Dataset A using the training configuration."""
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,help='Full Dataset A data.yaml')
    p.add_argument('--device',default=None,help='Optional local device, e.g. cpu, mps, 0')
    p.add_argument('--initial-weights',default='yolov8n.pt')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    cfg=yaml.safe_load((root/'code/training/training_config.yaml').read_text())
    cfg.update(data=str(Path(a.data).resolve()),project=str(root/'results/training'),name='detector_training')
    if a.device is not None:cfg['device']=a.device
    YOLO(a.initial_weights).train(**cfg)
if __name__=='__main__':main()
