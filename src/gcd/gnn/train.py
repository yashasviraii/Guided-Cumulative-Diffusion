"""CLI: train the entity-priority GCN (or MLP ablation) on constructed graphs.

Usage
-----
    python -m gcd.gnn.train \\
        --graph-dir data/gnn_graphs \\
        --model-type gcn \\
        --epochs 100

The graph directory must contain ``{split}/*.json`` files produced by
``gcd.data.graph_builder.SceneGraphBuilder`` (splits: "train", "test").
"""

from __future__ import annotations

import argparse

import torch
from torch_geometric.loader import DataLoader

from gcd.gnn.dataset import GraphDataset
from gcd.gnn.models import NodeLevelGNN, SimpleMLP
from gcd.gnn.trainer import GNNTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the GCN entity-priority predictor")
    parser.add_argument("--graph-dir", default="data/gnn_graphs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model-type",
        choices=["gcn", "mlp"],
        default="gcn",
        help="'gcn' uses graph convolutions; 'mlp' is a fast node-wise ablation.",
    )
    parser.add_argument("--checkpoint-out", default="checkpoints/gnn_model.pt")
    parser.add_argument("--plot-out", default="checkpoints/gnn_training_history.png")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(f"Using device: {args.device}")

    train_dataset = GraphDataset(args.graph_dir, split="train")
    test_dataset = GraphDataset(args.graph_dir, split="test")

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    in_channels = next(iter(train_loader)).x.shape[1]
    print(f"Input feature dimension: {in_channels}")

    if args.model_type == "gcn":
        model: torch.nn.Module = NodeLevelGNN(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
        )
    else:
        print("Using SimpleMLP for fast, structure-free training.")
        model = SimpleMLP(in_channels)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = GNNTrainer(
        model,
        device=args.device,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_out,
    )
    trainer.fit(train_loader, val_loader, test_loader, epochs=args.epochs, patience=args.patience)
    trainer.plot_history(args.plot_out)
    print(f"\nDone. Best checkpoint saved to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
