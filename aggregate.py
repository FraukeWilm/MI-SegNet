import os
import json
import numpy as np
import argparse 
import pandas as pd

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", help="Define experiment directory.")
    args = parser.parse_args()
    ious = []
    runs = filter(lambda file: file.__contains__('fold'), os.listdir(args.experiment_dir))
    for run in list(runs):
        with open(os.path.join(args.experiment_dir, run, 'files', "wandb-metadata.json"), 'r') as stream:
            config = json.load(stream)
        with open(os.path.join(args.experiment_dir, run, 'files', "iou.json"), 'r') as stream:
            iou = json.load(stream)
        iou = {key: np.round(float(iou[key]['mIoU']),2) for key in iou.keys()}
        iou['loss'] = config['args'][1].split('config_')[-1]
        iou['network'] = config['args'][5]
        iou['fold'] = config['args'][7]
        ious.append(iou)
    df = pd.DataFrame(ious, columns=['network', 'loss', 'fold', 'cs2', 'nz20', 'nz210', 'gt450', 'p1000'])
    df.to_csv(os.path.join(args.experiment_dir, 'mIoUs.csv'), index=False)
        



