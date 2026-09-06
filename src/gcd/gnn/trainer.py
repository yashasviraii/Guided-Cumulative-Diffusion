"""Training loop for the entity-priority GCN/MLP (Section 3.3)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class GNNTrainer:
    """Trains a node-level priority regressor with MSE loss and early stopping."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        checkpoint_path: str = "best_model.pt",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index)
            loss = self.criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, name: str = "Val") -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_pred, all_true = [], []
        for batch in tqdm(loader, desc=name, leave=False):
            batch = batch.to(self.device)
            out = self.model(batch.x, batch.edge_index)
            total_loss += self.criterion(out, batch.y).item()
            all_pred.extend(out.cpu().numpy())
            all_true.extend(batch.y.cpu().numpy())
        mae = float(np.mean(np.abs(np.array(all_pred) - np.array(all_true))))
        mse = total_loss / max(1, len(loader))
        return mse, mae

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        epochs: int = 100,
        patience: int = 20,
    ) -> None:
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.evaluate(val_loader, "Val")
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % 5 == 0:
                print(
                    f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(torch.load(self.checkpoint_path))

        if test_loader is not None:
            test_loss, test_mae = self.evaluate(test_loader, "Test")
            print(f"\nTest Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

    def plot_history(self, save_path: str = "gnn_training_history.png") -> None:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for ax, log_scale in ((ax1, False), (ax2, True)):
            ax.plot(self.history["train_loss"], label="Train", marker="o", markersize=3)
            ax.plot(self.history["val_loss"], label="Val", marker="s", markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss" + (" (log scale)" if log_scale else ""))
            ax.set_title("Training and Validation Loss" + (" (Log Scale)" if log_scale else ""))
            if log_scale:
                ax.set_yscale("log")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved training curve to {save_path}")
