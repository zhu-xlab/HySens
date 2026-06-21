import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# models
from models.hysens_encoder import HysensEncoder
from models.hysens_model import HysensPretrainModel
from models.spectral_embedding import WavelengthSpectralEmbedding
from models.spectral_pe import SpectralPositionalEncoding
from models.spectral_encoder import SpectralEncoder
from models.spectral_decoder import SpectralDecoder
from models.spectral_harmonizer import SpectralHarmonizer

# dataset
from datasets.hyspecnet import HyspecnetDataset, load_enmap_wavelengths

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.set_num_threads(16)


# -----------------------
# CONFIG
# -----------------------
class Config:
    data_dir = "./data/preprocess_patches"

    log_dir = "./logs/hysens_pretrain"
    save_dir = "./checkpoints/hysens"

    batch_size = 1
    num_workers = 2

    lr = 1e-4
    epochs = 50
    mask_ratio = 0.8

    device = "cuda" if torch.cuda.is_available() else "cpu"


def l1_loss(pred, target):
    pred = pred.squeeze()
    target = target.squeeze()
    loss = torch.abs(pred - target)
    loss = torch.mean(loss)
    return loss


# -----------------------
# MAIN
# -----------------------
def main():

    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    writer = SummaryWriter(cfg.log_dir)

    # -----------------------
    # WAVELENGTHS
    # -----------------------
    source_wl = load_enmap_wavelengths()
    target_wl = np.arange(400, 2500 + 1, 10, dtype=np.float32)[:210]

    harmonizer = SpectralHarmonizer(source_wl, target_wl)

    # -----------------------
    # DATASET
    # -----------------------
    dataset = HyspecnetDataset(cfg.data_dir, harmonizer=harmonizer)

    train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=1, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=2, pin_memory=True, shuffle=False, drop_last=True)

    # -----------------------
    # MODEL
    # -----------------------
    input_dim = 210
    block_size = 3
    hidden_dim = 96
    num_blocks = input_dim // block_size

    embedding = WavelengthSpectralEmbedding(input_dim=input_dim, block_size=block_size, hidden_dim=hidden_dim)

    pe = SpectralPositionalEncoding(dim=96)

    encoder_core = SpectralEncoder(hidden_dim=hidden_dim, n_heads=8, n_layers=12)

    encoder = HysensEncoder(embedding_layer=embedding, pe=pe, encoder=encoder_core,
                            hidden_dim=hidden_dim, block_size=block_size, num_blocks=num_blocks)

    decoder = SpectralDecoder(input_dim=input_dim, block_size=block_size, hidden_dim=hidden_dim)

    model = HysensPretrainModel(encoder=encoder, decoder=decoder).to(cfg.device)

    # -----------------------
    # OPTIMIZER
    # -----------------------
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    global_step = 0

    criterion = l1_loss

    # -----------------------
    # TRAINING LOOP
    # -----------------------
    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0

        # Training Loop
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit='batch') as train_bar:
            for pixels in train_bar:
                pixels = pixels.to(cfg.device)
                num_pixels = pixels.size(2)
                shuffled_indices = torch.randperm(num_pixels)
                subpixel_indices = shuffled_indices.split(4096)

                for subpixel_indice in subpixel_indices:
                    optimizer.zero_grad()
                    subpixel = pixels[:, :, subpixel_indice]
                    # Forward pass
                    # mask_ratio = random.uniform(0.5, 0.9)
                    output = model(subpixel, mask_ratio=0.8)

                    # Compute loss
                    loss = criterion(output, subpixel)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_train_loss += loss.item()

                    # Log the training loss per batch to TensorBoard
                    writer.add_scalar('Loss/Train_batch', loss.item(), global_step)
                    global_step += 1

                    # Update progress bar description
                    train_bar.set_postfix({'Train Loss': loss.item()})

        avg_train_loss = total_train_loss / (len(train_loader) * len(subpixel_indices))

        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Validation {epoch + 1}/{cfg.epochs}', unit='batch') as val_bar:
                for pixels in val_bar:
                    pixels = pixels.to(cfg.device)

                    num_pixels = pixels.size(2)
                    shuffled_indices = torch.randperm(num_pixels)
                    subpixel_indices = shuffled_indices.split(4096)

                    for subpixel_indice in subpixel_indices:
                        subpixel = pixels[:, :, subpixel_indice]
                        # mask_ratio = random.uniform(0.5, 0.9)
                        output = model(subpixel, mask_ratio=0.8)
                        loss = criterion(output, subpixel)
                        total_val_loss += loss.item()

                        val_bar.set_postfix({'Val Loss': loss.item()})

        avg_val_loss = total_val_loss / (len(val_loader) * len(subpixel_indices))

        # Log the average training and validation loss per epoch to TensorBoard
        writer.add_scalar('Loss/Train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val_epoch', avg_val_loss, epoch)

        # Save the model every 1 epoch
        if (epoch + 1) % 1 == 0:
            model_save_path = os.path.join(cfg.save_dir, f'hysens_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1} to {model_save_path}")

    # Save the final model after training
    final_model_save_path = os.path.join(cfg.save_dir, 'hysens_final.pth')
    torch.save(model.state_dict(), final_model_save_path)
    print(f"Final model saved to {final_model_save_path}")

    writer.close()


if __name__ == "__main__":
    main()