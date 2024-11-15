import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import *
from tqdm import tqdm

from models import RGB2GreyUNet


def train_model(
    train_ds: Dataset,
    val_ds: Dataset,
    fr_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    learning_rate: float,
    epochs: int,
    batch_size: int,
    device: torch.device, 
) -> Tuple[RGB2GreyUNet, Dict[str, List[float]]]:
    model = RGB2GreyUNet()
    clf_layer = nn.Linear(1024, 10177) # 10177 classes in CelebA
    model.to(device)
    clf_layer.to(device)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    
    reconstruction_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc')}
    for epoch in range(epochs):
        train_loss, train_acc = 0, 0
        model.train()
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}")
        for i, batch in enumerate(pbar):
            imgs, greys, id_label = batch
            imgs, greys, id_label = imgs.to(device), greys.to(device), id_label.to(device)
            
            optimizer.zero_grad()
            
            outputs, emb = model(imgs)
            pred_ids = clf_layer(emb)
            
            fr_loss = fr_loss_fn(pred_ids, id_label)
            reconstruction_loss = reconstruction_loss_fn(outputs, greys)
            loss = fr_loss + reconstruction_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (pred_ids.argmax(dim=1) == id_label).sum().item()
            
            pbar.set_postfix_str(
                f'Train Loss: {train_loss / (i + 1):.4f}, Train Accuracy: {100 * (train_acc / (i + 1)):.4f}%'
            )
        
        train_loss /= len(train_dl)
        train_acc /= len(train_dl)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
    
    
        val_loss, val_acc = 0, 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dl):
                imgs, greys, id_label = batch
                imgs, greys, id_label = imgs.to(device), greys.to(device), id_label.to(device)
                
                outputs, emb = model(imgs)
                pred_ids = clf_layer(emb)
                
                fr_loss = fr_loss_fn(pred_ids, id_label)
                reconstruction_loss = reconstruction_loss_fn(outputs, greys)
                loss = fr_loss + reconstruction_loss
                
                val_loss += loss.item()
                val_acc += (pred_ids.argmax(dim=1) == id_label).sum().item()
                
                print(f'Val Loss: {val_loss / (i + 1):.4f}, Val Accuracy: {100 * (val_acc / (i + 1)):.4f}%')
        
        val_loss /= len(val_dl)
        val_acc /= len(val_dl)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    return model, history
