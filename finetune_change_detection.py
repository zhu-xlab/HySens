import os
import random
import numpy as np
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

# models
from models.hysens_encoder import HysensEncoder_FT
from models.hysens_model import HysensChangeDetectionModel
from models.spectral_embedding import WavelengthSpectralEmbedding_FT
from models.spectral_pe import SpectralPositionalEncoding_FT
from models.spectral_encoder import SpectralEncoder
from models.spectral_harmonizer import SpectralHarmonizer_FT
from models.spectral_spatial_aggregator import Spectral_Spatial_Aggregator

# dataset
from datasets.registry import get_dataset, get_wavelength_loader, get_dataloader

from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.set_num_threads(16)


# -----------------------
# CONFIG
# -----------------------
class Config:
    # -----------------------
    # dataset switch
    # -----------------------
    dataset = "hermiston"  # hermiston / bay / barbara

    # base root
    data_root = "./data_downstream"

    # dataset-specific mapping
    dataset_map = {
        "hermiston": {
            "data_dir_pre": "Hermiston/hermiston2004.mat",
            "data_dir_post": "Hermiston/hermiston2007.mat",
            "label_dir": "Hermiston/rdChangesHermiston_5classes.mat",
        },
        "bay": {
            "data_dir_pre": "BayArea/Bay_Area_2013.mat",
            "data_dir_post": "BayArea/Bay_Area_2015.mat",
            "label_dir": "BayArea/bayArea_gtChanges2.mat",
        },
        "barbara": {
            "data_dir_pre": "Barbara/barbara_2013.mat",
            "data_dir_post": "Barbara/barbara_2014.mat",
            "label_dir": "Barbara/barbara_gtChanges.mat",
        }
    }

    log_dir = "./logs/change_detection"
    ckpt_dir = "./checkpoints/hysens/hysens_final.pth"

    patch_size = 8
    batch_size = 16
    num_workers = 2

    lr = 2e-5
    epochs = 500
    patience = 20

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
            os.path.join(cls.data_root, cfg["data_dir_pre"]),
            os.path.join(cls.data_root, cfg["data_dir_post"]),
            os.path.join(cls.data_root, cfg["label_dir"]),
        )


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        hyper_patch_pre, hyper_patch_post, labels = batch
        hyper_patch_pre, hyper_patch_post, labels = hyper_patch_pre.to(device), hyper_patch_post.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(hyper_patch_pre, hyper_patch_post)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    num_classes = 2
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            hyper_patch_pre, hyper_patch_post, labels = batch
            hyper_patch_pre, hyper_patch_post, labels = hyper_patch_pre.to(device), hyper_patch_post.to(
                device), labels.to(device)
            outputs = model(hyper_patch_pre, hyper_patch_post)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # get class predictions
            _, predicted = torch.max(outputs, 1)
            # Update per-class accuracy
            for i in range(labels.size(0)):  # Iterate over batch
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Collect all predictions and labels for Kappa calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    overall_accuracy = correct_predictions / total_samples

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_predictions)
    # Precision / Recall / F1 (macro is standard for CD)
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    return avg_loss, overall_accuracy, kappa, precision, recall, f1


def main_training_loop(model, train_loader, val_loader, test_loader,
                       criterion, optimizer, num_epochs, patience, device):
    os.makedirs(Config.log_dir, exist_ok=True)
    writer = SummaryWriter(Config.log_dir)

    # Initialize variables to track the best validation OA and corresponding epoch
    best_val_oa = -np.inf
    best_val_loss = np.inf
    best_epoch = -1
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        writer.add_scalar('Training Loss', train_loss, epoch)
        val_loss, val_oa, val_kp, val_pre, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Overall Accuracy', val_oa, epoch)
        writer.add_scalar('Validation Kappa', val_kp, epoch)
        writer.add_scalar('Validation Precision', val_pre, epoch)
        writer.add_scalar('Validation Recall', val_rec, epoch)
        writer.add_scalar('Validation F1', val_f1, epoch)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f'Validation Loss: {val_loss:.4f}, Validation OA: {val_oa:.4f}')

        # Save the best model based on validation OA
        if val_oa > best_val_oa:
            best_val_oa = val_oa
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0  # Reset the counter
        else:
            epochs_without_improvement += 1  # Increment the counter

        # Early stopping if no improvement for 'patience' consecutive epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_wts)

    # Final evaluation on the test set with the best model
    print(f"Best model found at epoch {best_epoch + 1} :")
    val_loss, val_oa, val_kp, val_pre, val_rec, val_f1 = validate(model, val_loader, criterion, device)
    test_loss, test_oa, test_kp, test_pre, test_rec, test_f1 = validate(model, test_loader, criterion, device)
    print(
        f'Final Val Loss: {val_loss:.4f}, Final Val OA: {val_oa:.4f}, '
        f'Final Val Kappa: {val_kp:.4f}, '
        f'Final Val Precision: {val_pre:.4f}, Final Val Recall: {val_rec:.4f}, Final Val F1: {val_f1:.4f}'
    )
    print(
        f'Final Test Loss: {test_loss:.4f}, Final Test OA: {test_oa:.4f}, '
        f'Final Test Kappa: {test_kp:.4f}, '
        f'Final Test Precision: {test_pre:.4f}, Final Test Recall: {test_rec:.4f}, Final Test F1: {test_f1:.4f}'
    )

    print("Training Completed")
    writer.close()
    return test_loss, test_oa, test_kp, test_pre, test_rec, test_f1


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
        data_dir_pre, data_dir_post, label_dir = cfg.get_paths()

        # -----------------------
        # WAVELENGTHS
        # -----------------------
        dataset_change = get_dataset(cfg.dataset)
        load_wavelengths = get_wavelength_loader(cfg.dataset)
        create_loaders = get_dataloader(cfg.dataset)

        source_wl = load_wavelengths()
        target_wl = np.arange(400, 2500 + 1, 10, dtype=np.float32)[:210]

        harmonizer = SpectralHarmonizer_FT(source_wl, target_wl)

        # -----------------------
        # DATASET
        # -----------------------
        dataset = dataset_change(data_dir_pre=data_dir_pre, data_dir_post=data_dir_post, label_dir=label_dir,
                                 harmonizer=harmonizer, patch_size=cfg.patch_size)
        train_loader, val_loader, test_loader = create_loaders(dataset, batch_size=cfg.batch_size)

        # -----------------------
        # MODEL
        # -----------------------
        ref_dim = 210
        block_size = 3
        hidden_dim = 96
        ref_num_blocks = 70
        input_dim = dataset.image_pre.shape[2]
        num_blocks = input_dim // 3

        group_indices = (harmonizer.selected_target_indices[0:-1:3] / 3).astype(int)
        embedding = WavelengthSpectralEmbedding_FT(ref_dim=ref_dim, input_dim=input_dim, block_size=block_size,
                                                   hidden_dim=hidden_dim, wavelengths_group_indices=group_indices)

        pe = SpectralPositionalEncoding_FT(dim=hidden_dim)

        encoder_core = SpectralEncoder(hidden_dim=hidden_dim, n_heads=8, n_layers=12)

        encoder = HysensEncoder_FT(embedding_layer=embedding, pe=pe, encoder=encoder_core,
                                hidden_dim=hidden_dim, block_size=block_size, num_blocks=ref_num_blocks,
                                wavelengths_group_indices=group_indices)

        aggregator = Spectral_Spatial_Aggregator()

        model = HysensChangeDetectionModel(encoder=encoder, aggregator=aggregator,
                                           batch_size=cfg.batch_size,
                                          patch_size=cfg.patch_size, num_blocks=num_blocks,
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

        criterion = nn.CrossEntropyLoss()

        test_loss, test_oa, test_kp, test_pre, test_rec, test_f1 = (
            main_training_loop(model, train_loader, val_loader,
                               test_loader, criterion, optimizer, cfg.epochs, cfg.patience, cfg.device))
        results.append({"run": i + 1, "loss": test_loss, "oa": test_oa, "kappa": test_kp,
                        "precision": test_pre, "recall": test_rec, "F1": test_f1})

    df = pd.DataFrame(results, columns=["run", "loss", "oa", "kappa", "precision", "recall", "F1"])

    # Compute mean and std for each metric
    mean_row = df[["loss", "oa", "kappa", "precision", "recall", "F1"]].mean()
    mean_row["run"] = "mean"

    std_row = df[["loss", "oa", "kappa", "precision", "recall", "F1"]].std()
    std_row["run"] = "std"

    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(df)


if __name__ == "__main__":
    run_experiments(num_runs=5, base_seed=42, output_file=os.path.join(Config().log_dir, "results.csv"))