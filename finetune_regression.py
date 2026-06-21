import os
import random
import numpy as np
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

# models
from models.hysens_encoder import HysensEncoder_FT_Pixel
from models.hysens_model import HysensRegressionModel
from models.spectral_embedding import WavelengthSpectralEmbedding_FT
from models.spectral_pe import SpectralPositionalEncoding_FT
from models.spectral_encoder import SpectralEncoder
from models.spectral_harmonizer import SpectralHarmonizer_FT

# dataset
from datasets.registry import get_dataset, get_wavelength_loader, get_dataloader

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
torch.set_num_threads(16)


# -----------------------
# CONFIG
# -----------------------
class Config:
    # -----------------------
    # dataset switch
    # -----------------------
    dataset = "karly"  # gloria / karly

    # base root
    data_root = "./data_downstream"

    # dataset-specific mapping
    dataset_map = {
        "gloria": {
            "data_dir": "GLORIA_2022/GLORIA_Rrs.csv",
            "label_dir": "GLORIA_2022/GLORIA_meta_and_lab.csv",
        },
        "karly": {
            "data_dir": "Moisture/soilmoisture_dataset.csv",
            "label_dir": "",
        }
    }

    log_dir = "./logs/regression"
    ckpt_dir = "./checkpoints/hysens/hysens_final.pth"

    batch_size = 16
    num_workers = 2

    lr = 2e-5
    epochs = 500
    patience = 100

    # -----------------------------------
    # tuning strategy
    # scratch
    # linear_probing
    # freeze_embedding
    # full_finetuning
    # -----------------------------------
    tuning_mode = "full_finetuning"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_paths(cls):
        cfg = cls.dataset_map[cls.dataset]
        return (
            os.path.join(cls.data_root, cfg["data_dir"]),
            os.path.join(cls.data_root, cfg["label_dir"]),
        )


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        hyper_pixel, labels = batch
        hyper_pixel, labels = hyper_pixel.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(hyper_pixel)
        outputs = outputs.squeeze()

        loss = criterion(outputs, labels)
        loss.backward()

        max_norm = 1.0  # gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for batch in val_loader:
            hyper_pixel, labels = batch
            hyper_pixel, labels = hyper_pixel.to(device), labels.to(device)
            outputs = model(hyper_pixel)
            outputs = outputs.squeeze()
            y_true += labels.cpu().numpy().tolist()
            y_pred += outputs.cpu().numpy().tolist()
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r_square = r2_score(y_true, y_pred)
        loss = criterion(outputs, labels)
    return loss, rmse, r_square


def main_training_loop(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, patience, device):
    os.makedirs(Config.log_dir, exist_ok=True)
    writer = SummaryWriter(Config.log_dir)

    best_val_rmse = np.inf
    best_val_loss = np.inf
    best_val_r2 = -np.inf
    best_epoch = -1
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        writer.add_scalar('Training Loss', train_loss, epoch)
        val_loss, val_rmse, val_r2 = validate(model, val_loader, criterion, device)
        test_loss, test_rmse, test_r2 = validate(model, test_loader, criterion, device)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation RMSE', val_rmse, epoch)
        writer.add_scalar('Validation R2', val_r2, epoch)
        writer.add_scalar('Test RMSE', test_rmse, epoch)
        writer.add_scalar('Test R2', test_r2, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f'Validation Loss: {val_loss:.4f}, Validation RMSE: {val_rmse:.4f}, Validation R2: {val_r2:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}')

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_wts)

    # Final evaluation on the test set with the best model
    print(f"Best model found at epoch {best_epoch + 1} :")
    val_loss, val_rmse, val_r2 = validate(model, val_loader, criterion, device)
    test_loss, test_rmse, test_r2 = validate(model, test_loader, criterion, device)
    print(
        f'Final Val Loss: {val_loss:.4f}, Final Val RMSE: {val_rmse:.4f}, Final Val R2: {val_r2:.4f}, ')
    print(
        f'Final Test Loss: {test_loss:.4f}, Final Test RMSE: {test_rmse:.4f}, Final Test R2: {test_r2:.4f}, ')

    print("Training Completed")
    writer.close()
    return test_loss, test_rmse, test_r2


def apply_tuning_strategy(model, tuning_mode):

    # reset everything to trainable
    for p in model.parameters():
        p.requires_grad = True

    if tuning_mode == "linear_probing":
        for p in model.encoder.parameters():
            p.requires_grad = False

    elif tuning_mode == "freeze_embedding":
        for p in model.encoder.embedding.parameters():
            p.requires_grad = False

    elif tuning_mode in ["scratch", "full_finetuning"]:
        pass

    else:
        raise ValueError(f"Unknown tuning mode: {tuning_mode}")


def run_experiments(num_runs: int, base_seed: int = 42, output_file: str = "results.csv"):
    results = []

    for i in range(num_runs):
        # fix the seed for reproduce
        seed = base_seed + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # single run
        cfg = Config()
        data_dir, label_dir = cfg.get_paths()

        # -----------------------
        # WAVELENGTHS
        # -----------------------
        dataset_regression = get_dataset(cfg.dataset)
        load_wavelengths = get_wavelength_loader(cfg.dataset)
        create_loaders = get_dataloader(cfg.dataset)

        source_wl = load_wavelengths()
        target_wl = np.arange(400, 2500 + 1, 10, dtype=np.float32)[:210]

        harmonizer = SpectralHarmonizer_FT(source_wl, target_wl)

        # -----------------------
        # DATASET
        # -----------------------
        dataset = dataset_regression(data_dir=data_dir, label_dir=label_dir, harmonizer=harmonizer)
        train_loader, val_loader, test_loader = create_loaders(dataset, batch_size=cfg.batch_size)

        # -----------------------
        # MODEL
        # -----------------------
        ref_dim = 210
        block_size = 3
        hidden_dim = 96
        ref_num_blocks = 70
        input_dim = dataset.pixels.shape[1]
        num_blocks = input_dim // 3

        group_indices = (harmonizer.selected_target_indices[0:-1:3] / 3).astype(int)
        embedding = WavelengthSpectralEmbedding_FT(ref_dim=ref_dim, input_dim=input_dim, block_size=block_size,
                                                   hidden_dim=hidden_dim, wavelengths_group_indices=group_indices)

        pe = SpectralPositionalEncoding_FT(dim=hidden_dim)

        encoder_core = SpectralEncoder(hidden_dim=hidden_dim, n_heads=8, n_layers=12)

        encoder = HysensEncoder_FT_Pixel(embedding_layer=embedding, pe=pe, encoder=encoder_core,
                                hidden_dim=hidden_dim, block_size=block_size, num_blocks=ref_num_blocks,
                                wavelengths_group_indices=group_indices)

        model = HysensRegressionModel(encoder=encoder, batch_size=cfg.batch_size,num_blocks=num_blocks,
                                      hidden_dim=hidden_dim, input_dim=input_dim).to(cfg.device)

        # ---------------------------------------
        # load pretrained weights if needed
        # ---------------------------------------
        if cfg.tuning_mode != "scratch":

            print(f"Loading pretrained checkpoint: {cfg.ckpt_dir}")

            check_point = torch.load(cfg.ckpt_dir)

            model_state_dict = model.state_dict()

            for name, param in check_point.items():
                if name in model_state_dict and param.shape == model_state_dict[name].shape:
                    model_state_dict[name].copy_(param)

            model.load_state_dict(model_state_dict)

        else:
            print("Training from scratch. No pretrained weights loaded.")

        apply_tuning_strategy(model, cfg.tuning_mode)

        # -----------------------
        # OPTIMIZER
        # -----------------------
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=2e-05,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decay=0.05
        )

        criterion = nn.MSELoss()

        test_loss, test_rmse, test_r2 = main_training_loop(model, train_loader, val_loader, test_loader,
                                                           criterion, optimizer, cfg.epochs, cfg.patience,
                                                           cfg.device)
        results.append({"run": i + 1, "loss": test_loss, "rmse": test_rmse, "r2": test_r2})

    df = pd.DataFrame(results, columns=["run", "loss", "rmse", "r2"])

    # Compute mean and std for each metric
    mean_row = df[["run", "loss", "rmse", "r2"]].mean()
    mean_row["run"] = "mean"

    std_row = df[["run", "loss", "rmse", "r2"]].std()
    std_row["run"] = "std"

    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(df)


if __name__ == "__main__":
    run_experiments(num_runs=5, base_seed=42, output_file=os.path.join(Config().log_dir, "results.csv"))