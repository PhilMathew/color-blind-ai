import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import *
from tqdm import tqdm
from sklearn.metrics import roc_curve

from models import RGB2GreyUNet, ColorblindEncoder


NUM_CELEBA_CLASSES = 10178


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
    clf_layer = nn.Linear(1024, NUM_CELEBA_CLASSES)
    model.to(device)
    clf_layer.to(device)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=16)
    
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
            
            if id_label.max().item() > NUM_CELEBA_CLASSES:
                raise ValueError(f'CelebA has at least {id_label.max().item()} classes; consider increasing NUM_CELEBA_CLASSES')
            
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
                if i > 5:
                    break
                imgs, greys, id_label = batch
                imgs, greys, id_label = imgs.to(device), greys.to(device), id_label.to(device)
                
                outputs, emb = model(imgs)
                pred_ids = clf_layer(emb)
                
                fr_loss = fr_loss_fn(pred_ids, id_label)
                reconstruction_loss = reconstruction_loss_fn(outputs, greys)
                loss = fr_loss + reconstruction_loss
                
                val_loss += loss.item()
                val_acc += (pred_ids.argmax(dim=1) == id_label).sum().item()
                
                
        
        val_loss /= len(val_dl)
        val_acc /= len(val_dl)
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {100 * val_acc:.4f}%')
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    return model, history


def test_model(
    model: ColorblindEncoder,
    test_ds: Dataset,
    batch_size: int,
    device: torch.device,
):
    model.eval()
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=16)
    
    with torch.no_grad():
        labels, preds = [], []
        for i, batch in enumerate(test_dl):
            img1, img2, label = batch
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            _, emb1 = model(img1)
            _, emb2 = model(img2)
            
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(emb1, emb2, dim=1)
            labels.extend(label.cpu().numpy())
            preds.extend(cos_sim.cpu().numpy())
        
        labels, preds = np.array(labels), np.array(preds)
        fpr, tpr, thresh = roc_curve(labels, preds)
        auc = np.trapz(tpr, fpr)
        
    return fpr, tpr, thresh, auc
