"""Compute detector metrics on the complete validation set."""
import argparse,json
from pathlib import Path
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,help='Full Dataset A data.yaml')
    p.add_argument('--device',default='cpu')
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    out=root/'results/recomputed/detection';out.mkdir(parents=True,exist_ok=True)
    r=YOLO(root/'models/detector.pt').val(data=str(Path(a.data).resolve()),split='val',conf=.001,iou=.6,device=a.device,plots=False,save_json=False,project=str(out),name='validation',exist_ok=True)
    d={'precision':float(r.box.mp),'recall':float(r.box.mr),'mAP50':float(r.box.map50),'mAP50-95':float(r.box.map),'per_class':[{'class':r.names[i],'precision':float(r.box.p[i]),'recall':float(r.box.r[i]),'F1':float(r.box.f1[i]),'AP50':float(r.box.ap50[i]),'AP50-95':float(r.box.ap[i])} for i in range(len(r.names))]}
    d['F1'] = sum(x['F1'] for x in d['per_class']) / len(d['per_class'])
    d['F1_definition'] = 'unweighted mean of per-class F1'
    (out/'metrics.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
if __name__=='__main__':main()
