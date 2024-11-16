from argparse import ArgumentParser
import torch
import json
from pathlib import Path

from datasets import LFWPairsDataset
from baseline_models import iresnet50
from train_utils import test_model
from plot_utils import *


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/', help='/path/to/download/data/to/')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size for testing')
    parser.add_argument('-o', '--output_dir', default='./results/', help='/path/to/output/results/')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    test_ds = LFWPairsDataset(data_dir, img_size=(112, 112))
    
    model = iresnet50()
    with open('baseline_model.pth', 'rb') as f:
        model.load_state_dict(torch.load(f, weights_only=True))
    model.to(device)
    
    fpr, tpr, thresh, auc = test_model(
        model=model,
        test_ds=test_ds,
        batch_size=args.batch_size,
        device=device
    )

    # Save out args
    with open(str(output_dir / 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Save out test metrics
    with open(str(output_dir / 'test_metrics.json'), 'w') as f:
        json.dump(
            {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresh': thresh.tolist(),
                'auc': auc
            }, 
            f, 
            indent=4
        )
    plot_roc_curve(fpr, tpr, str(output_dir / 'roc_curve.png'))
    


if __name__ == '__main__':
    main()
