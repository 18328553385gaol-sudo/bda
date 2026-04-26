import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from autoencoder import PlayerAutoEncoder
from config import (
    DEFAULT_FEATURE_SOURCE_MODE,
    EPOCHS,
    FEATURE_COLS,
    HIDDEN_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    MODEL_PATH,
    PATIENCE,
    PROCESSED_FEATURES_PATH,
    TRAIN_BQ_FEATURE_TABLE,
    TRAIN_BQ_PROJECT,
)
from data_loader import load_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train player autoencoder from local parquet or BigQuery features."
    )
    parser.add_argument(
        "--source-mode",
        default=DEFAULT_FEATURE_SOURCE_MODE,
        choices=["local", "bigquery"],
        help="Feature source mode for training input.",
    )
    parser.add_argument(
        "--local-path",
        default=PROCESSED_FEATURES_PATH,
        help="Local parquet path when source mode is local.",
    )
    parser.add_argument(
        "--bq-project",
        default=TRAIN_BQ_PROJECT,
        help="BigQuery project ID when source mode is bigquery.",
    )
    parser.add_argument(
        "--bq-table",
        default=TRAIN_BQ_FEATURE_TABLE,
        help="BigQuery table name when source mode is bigquery.",
    )
    parser.add_argument(
        "--bq-limit",
        type=int,
        default=None,
        help="Optional row limit for BigQuery feature loading.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    return parser.parse_args()


def prepare_dataloaders(df, batch_size=64, val_ratio=0.2):
    features = df[FEATURE_COLS].values.astype(np.float32)
    tensor_x = torch.tensor(features)

    dataset = TensorDataset(tensor_x, tensor_x)

    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tensor_x


def train_autoencoder(
    model,
    train_loader,
    val_loader,
    epochs=30,
    lr=1e-3,
    device="cpu"
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    patience_counter = 0

    model.to(device)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * x_batch.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * x_batch.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {epoch_train_loss:.6f} | "
            f"Val Loss: {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    return train_losses, val_losses


def generate_embeddings(model, tensor_x, device="cpu"):
    model.to(device)
    model.eval()

    with torch.no_grad():
        x = tensor_x.to(device)
        embeddings = model.encoder(x).cpu().numpy()

    return embeddings


def save_embedding_artifacts(df, embeddings, latent_dim):
    os.makedirs("artifacts/embeddings", exist_ok=True)

    np.save(f"artifacts/embeddings/player_embeddings_{latent_dim}d.npy", embeddings)

    embedding_df = df[["player_key", "player_name", "position_group"]].copy()

    for i in range(embeddings.shape[1]):
        embedding_df[f"embedding_{i + 1}"] = embeddings[:, i]

    embedding_df.to_parquet(
        f"artifacts/embeddings/player_embedding_table_{latent_dim}d.parquet",
        index=False
    )

    print(f"Saved embeddings to artifacts/embeddings/ ({latent_dim}d)")


def save_training_history(train_losses, val_losses):
    os.makedirs("artifacts/training", exist_ok=True)

    history_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(train_losses) + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    )
    history_csv_path = f"artifacts/training/autoencoder_loss_history_{LATENT_DIM}d.csv"
    history_df.to_csv(history_csv_path, index=False)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["epoch"],
            y=history_df["train_loss"],
            mode="lines+markers",
            name="Train Loss",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history_df["epoch"],
            y=history_df["val_loss"],
            mode="lines+markers",
            name="Validation Loss",
        )
    )
    fig.update_layout(
        title=f"Autoencoder Training History ({LATENT_DIM}d)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
    )

    history_html_path = f"artifacts/training/autoencoder_loss_history_{LATENT_DIM}d.html"
    fig.write_html(history_html_path, include_plotlyjs="cdn")

    history_png_path = f"artifacts/training/autoencoder_loss_history_{LATENT_DIM}d.png"
    try:
        fig.write_image(history_png_path)
        print(f"Saved training curve image to {history_png_path}")
    except Exception as exc:
        print("Unable to save PNG training curve automatically.")
        print("Install 'kaleido' if you want static image export support.")
        print(f"PNG export error: {exc}")

    print(f"Saved training history to {history_csv_path}")
    print(f"Saved training curve to {history_html_path}")


def preprocess_training_features(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    for column in FEATURE_COLS:
        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")

    missing_rows = int(cleaned_df[FEATURE_COLS].isna().any(axis=1).sum())
    if missing_rows > 0:
        print("Missing values detected in training features. Applying fillna(0) to match notebook preprocessing.")
        print(cleaned_df[FEATURE_COLS].isna().sum()[cleaned_df[FEATURE_COLS].isna().sum() > 0].to_string())
        print(f"Rows with at least one missing feature: {missing_rows}")

    cleaned_df[FEATURE_COLS] = cleaned_df[FEATURE_COLS].fillna(0)

    for column in FEATURE_COLS:
        group_mean = cleaned_df.groupby("position_group")[column].transform("mean")
        group_std = cleaned_df.groupby("position_group")[column].transform("std")
        group_std = group_std.fillna(1).replace(0, 1)
        cleaned_df[column] = (cleaned_df[column] - group_mean) / group_std

    if np.isnan(cleaned_df[FEATURE_COLS].to_numpy()).any():
        raise ValueError("NaN values remain in training features after preprocessing.")

    return cleaned_df


def validate_training_columns(df: pd.DataFrame):
    required_columns = ["player_key", "player_name", "position_group", *FEATURE_COLS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required training columns: {', '.join(missing_columns)}")


def main():
    args = parse_args()

    df = load_features(
        source_mode=args.source_mode,
        local_path=args.local_path,
        bq_project=args.bq_project,
        bq_table=args.bq_table,
        bq_limit=args.bq_limit,
    )
    validate_training_columns(df)
    df = preprocess_training_features(df)

    print(f"Data loaded from {args.source_mode}: {df.shape}")

    train_loader, val_loader, tensor_x = prepare_dataloaders(
        df=df,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )

    model = PlayerAutoEncoder(
        input_dim=len(FEATURE_COLS),
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_losses, val_losses = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        device=device,
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    embeddings = generate_embeddings(model, tensor_x, device=device)
    print("Embedding shape:", embeddings.shape)

    save_embedding_artifacts(df, embeddings, LATENT_DIM)
    save_training_history(train_losses, val_losses)

    if train_losses and val_losses:
        print(f"Best validation loss: {min(val_losses):.6f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
