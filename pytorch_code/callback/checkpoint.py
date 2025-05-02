import os
import torch

def save_model(model, output_dir, epoch=None):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"model_epoch{epoch}.pth" if epoch is not None else "best_model.pth"
    path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), path)
