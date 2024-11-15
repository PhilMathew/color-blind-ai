from argparse import ArgumentParser
import torch
import json
from pathlib import Path

from datasets import CelebADataset, LFWPairsDataset
from losses import ArcFace
from train_utils import train_model, test_model
from plot_utils import *


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/', help='/path/to/download/data/to/')
    parser.add_argument('-l', '--loss_fn', default='crossentropy', help='loss function to use in training')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('-ne', '--num_epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('-o', '--output_dir', default='./results/', help='/path/to/output/results/')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    train_ds, val_ds = CelebADataset(str(data_dir), split='train'), CelebADataset(str(data_dir), split='valid')
    test_ds = LFWPairsDataset(data_dir)
    
    match args.loss_fn:
        case 'crossentropy':
            fr_loss_fn = torch.nn.CrossEntropyLoss()
        case 'arcface':
            fr_loss_fn = ArcFace()
        case _:
            raise ValueError(f"Invalid loss function: {args.loss_fn}")
    
    model, history = train_model(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.num_epochs,
        fr_loss_fn=fr_loss_fn,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device
    )
    fpr, tpr, thresh, auc = test_model(
        model=model.encoder,
        test_ds=test_ds,
        batch_size=args.batch_size,
        device=device
    )

    # Save out args
    with open(str(output_dir / 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Save out model weights
    torch.save(model.state_dict(), str(output_dir / 'model.pth'))
    torch.save(model.encoder.state_dict(), str(output_dir / 'encoder.pth'))
    
    # Save out training metrics
    with open(str(output_dir / 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    plot_history(history, output_dir / 'train_history.png')
    
    # Save out test metrics
    with open(str(output_dir / 'test_metrics.json'), 'w') as f:
        json.dump(
            {
                'fpr': fpr,
                'tpr': tpr,
                'thresh': thresh,
                'auc': auc
            }, 
            f, 
            indent=4
        )
    plot_roc_curve(fpr, tpr, str(output_dir / 'roc_curve.png'))
    


if __name__ == '__main__':
    main()
