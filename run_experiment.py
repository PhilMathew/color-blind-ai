from argparse import ArgumentParser
import torch
import json
from pathlib import Path

from datasets import CelebADataset
from train import train_model
from plot_utils import *


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/', help='/path/to/download/data/to/')
    parser.add_argument('-o', '--output_dir', default='.', help='/path/to/output/results/')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    train_ds, val_ds = CelebADataset(str(data_dir), split='train'), CelebADataset(str(data_dir), split='valid')
    
    model, history = train_model(
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=10,
        fr_loss_fn=torch.nn.CrossEntropyLoss(),
        batch_size=32,
        learning_rate=1e-3,
        device=device
    )

    # Save out necessary info
    torch.save(model.state_dict(), str(output_dir / 'model.pth'))
    torch.save(model.encoder.state_dict(), str(output_dir / 'encoder.pth'))
    with open(str(output_dir / 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    plot_history(history, output_dir / 'train_history.png')


if __name__ == '__main__':
    main()
