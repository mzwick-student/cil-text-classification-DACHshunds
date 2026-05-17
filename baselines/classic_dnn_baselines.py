from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from baselines.classic_ml_baselines import (
    ProgressTracker,
    StaticEmbeddingVectorizer,
    clean_text,
    load_data,
    score_predictions,
    tokenize_words,
)


N_CLASSES = 5
PAD_INDEX = 0
UNK_INDEX = 1


@dataclass(frozen=True)
class DnnExperimentSpec:
    experiment_id: str
    embedding_name: str
    input_type: str
    representation: str
    model_name: str
    report_rule: str

    @property
    def experiment(self) -> str:
        return (
            f"{self.experiment_id}__{self.embedding_name}"
            f"__{self.representation}__{self.model_name}"
        )


@dataclass(frozen=True)
class DnnExperimentResult:
    experiment: str
    experiment_id: str
    embedding: str
    input_type: str
    representation: str
    model: str
    report_rule: str
    status: str
    best_epoch: int | None = None
    train_seconds: float | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    mae: float | None = None
    cil_score: float | None = None
    quadratic_weighted_kappa: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class EpochResult:
    experiment: str
    epoch: int
    train_loss: float
    val_loss: float
    accuracy: float
    macro_f1: float
    mae: float
    cil_score: float
    quadratic_weighted_kappa: float


@dataclass
class EmbeddingResources:
    name: str
    path: str
    embeddings: dict[str, np.ndarray]
    dim: int
    idf: dict[str, float]
    term_counts: Counter
    token_coverage: float
    vocabulary_coverage: float


class DocumentVectorDataset(Dataset):
    def __init__(self, vectors: np.ndarray, labels: np.ndarray):
        self.vectors = torch.tensor(vectors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.vectors[idx], self.labels[idx]


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, lengths: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class TextCNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        num_filters: int,
        kernel_sizes: list[int],
        dropout: float,
        fine_tune_embeddings: bool,
    ):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            weights,
            freeze=not fine_tune_embeddings,
            padding_idx=PAD_INDEX,
        )
        dim = embedding_matrix.shape[1]
        self.convs = nn.ModuleList(
            [nn.Conv1d(dim, num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), N_CLASSES)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            features = torch.relu(conv(embedded))
            pooled.append(torch.max(features, dim=2).values)
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden_dim: int,
        dropout: float,
        fine_tune_embeddings: bool,
    ):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            weights,
            freeze=not fine_tune_embeddings,
            padding_idx=PAD_INDEX,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, N_CLASSES)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        final = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(final))


class SmallTransformerClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        max_length: int,
        num_layers: int,
        dropout: float,
        fine_tune_embeddings: bool,
    ):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            weights,
            freeze=not fine_tune_embeddings,
            padding_idx=PAD_INDEX,
        )
        self.position_embedding = nn.Embedding(max_length, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=choose_attention_heads(dim),
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, N_CLASSES)

    def forward(self, token_ids, lengths):
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        positions = positions.unsqueeze(0).expand_as(token_ids)
        padding_mask = token_ids.eq(PAD_INDEX)
        encoded = self.embedding(token_ids) + self.position_embedding(positions)
        encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
        non_pad = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))


def choose_attention_heads(dim: int) -> int:
    for heads in [8, 6, 5, 4, 3, 2]:
        if dim % heads == 0:
            return heads
    return 1


def experiment_specs(args: argparse.Namespace) -> list[DnnExperimentSpec]:
    specs = []
    if args.glove_path:
        specs.extend(
            [
                DnnExperimentSpec("D1", "glove", "document_vector", "average", "mlp", "XOR with D2"),
                DnnExperimentSpec("D2", "glove", "document_vector", "tfidf_weighted", "mlp", "best GloVe MLP"),
                DnnExperimentSpec("D5", "glove", "token_sequence", "pretrained_token_vectors", "textcnn", "compare to D7"),
                DnnExperimentSpec("D6", "glove", "token_sequence", "pretrained_token_vectors", "bilstm", "compare to D8"),
                DnnExperimentSpec("D10", "glove", "token_sequence", "pretrained_token_vectors", "small_transformer", "optional, lowest priority"),
            ]
        )
    if args.fasttext_path:
        specs.extend(
            [
                DnnExperimentSpec("D3", "fasttext", "document_vector", "average", "mlp", "XOR with D4"),
                DnnExperimentSpec("D4", "fasttext", "document_vector", "tfidf_weighted", "mlp", "best fastText MLP"),
                DnnExperimentSpec("D7", "fasttext", "token_sequence", "pretrained_token_vectors", "textcnn", "likely report"),
                DnnExperimentSpec("D8", "fasttext", "token_sequence", "pretrained_token_vectors", "bilstm", "likely report"),
                DnnExperimentSpec("D9", "fasttext", "token_sequence", "pretrained_token_vectors", "small_transformer", "optional diagnostic report"),
            ]
        )
    if not args.include_transformers:
        specs = [spec for spec in specs if spec.model_name != "small_transformer"]
    return specs


def embedding_paths(args: argparse.Namespace) -> dict[str, str]:
    paths = {}
    if args.glove_path:
        paths["glove"] = args.glove_path
    if args.fasttext_path:
        paths["fasttext"] = args.fasttext_path
    return paths


def prepare_embedding_resources(
    name: str,
    path: str,
    train_texts: pd.Series,
) -> EmbeddingResources:
    vectorizer = StaticEmbeddingVectorizer(embedding_path=path, strategy="average")
    vectorizer.fit(train_texts)
    return EmbeddingResources(
        name=name,
        path=path,
        embeddings=vectorizer.embeddings_,
        dim=vectorizer.dim_,
        idf=vectorizer.idf_,
        term_counts=vectorizer.term_counts_,
        token_coverage=vectorizer.coverage_["token_coverage"],
        vocabulary_coverage=vectorizer.coverage_["vocabulary_coverage"],
    )


def build_document_vectors(
    texts: pd.Series,
    resources: EmbeddingResources,
    strategy: str,
) -> np.ndarray:
    rows = []
    for text in texts:
        vectors = []
        weights = []
        for token in tokenize_words(clean_text(text)):
            vector = resources.embeddings.get(token)
            if vector is None:
                continue
            vectors.append(vector)
            if strategy == "tfidf_weighted":
                weights.append(resources.idf.get(token, 1.0))
            else:
                weights.append(1.0)
        if not vectors:
            rows.append(np.zeros(resources.dim, dtype=np.float32))
            continue
        matrix = np.vstack(vectors).astype(np.float32)
        weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        rows.append((matrix * weights_arr).sum(axis=0) / weights_arr.sum())
    return np.vstack(rows).astype(np.float32)


def build_sequence_inputs(
    train_texts: pd.Series,
    val_texts: pd.Series,
    resources: EmbeddingResources,
    max_length: int,
    max_sequence_vocab: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    covered_tokens = [
        token for token, _ in resources.term_counts.most_common()
        if token in resources.embeddings
    ]
    covered_tokens = covered_tokens[:max_sequence_vocab]

    token_to_idx = {token: idx + 2 for idx, token in enumerate(covered_tokens)}
    embedding_matrix = np.zeros((len(token_to_idx) + 2, resources.dim), dtype=np.float32)
    for token, idx in token_to_idx.items():
        embedding_matrix[idx] = resources.embeddings[token]

    train_sequences, train_lengths = encode_texts(train_texts, token_to_idx, max_length)
    val_sequences, val_lengths = encode_texts(val_texts, token_to_idx, max_length)
    return train_sequences, train_lengths, val_sequences, val_lengths, embedding_matrix


def encode_texts(
    texts: pd.Series,
    token_to_idx: dict[str, int],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences = np.full((len(texts), max_length), PAD_INDEX, dtype=np.int64)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for row_idx, text in enumerate(texts):
        ids = [token_to_idx.get(token, UNK_INDEX) for token in tokenize_words(clean_text(text))]
        ids = ids[:max_length]
        if not ids:
            ids = [UNK_INDEX]
        lengths[row_idx] = len(ids)
        sequences[row_idx, : len(ids)] = ids
    return sequences, lengths


def make_model(
    spec: DnnExperimentSpec,
    input_dim: int | None,
    embedding_matrix: np.ndarray | None,
    args: argparse.Namespace,
) -> nn.Module:
    if spec.model_name == "mlp":
        if input_dim is None:
            raise ValueError("MLP needs input_dim")
        return MLPClassifier(
            input_dim=input_dim,
            hidden_dim=args.mlp_hidden_dim,
            dropout=args.dropout,
        )
    if embedding_matrix is None:
        raise ValueError(f"{spec.model_name} needs embedding_matrix")
    if spec.model_name == "textcnn":
        return TextCNNClassifier(
            embedding_matrix=embedding_matrix,
            num_filters=args.cnn_num_filters,
            kernel_sizes=args.cnn_kernel_sizes,
            dropout=args.dropout,
            fine_tune_embeddings=args.fine_tune_embeddings,
        )
    if spec.model_name == "bilstm":
        return BiLSTMClassifier(
            embedding_matrix=embedding_matrix,
            hidden_dim=args.lstm_hidden_dim,
            dropout=args.dropout,
            fine_tune_embeddings=args.fine_tune_embeddings,
        )
    if spec.model_name == "small_transformer":
        return SmallTransformerClassifier(
            embedding_matrix=embedding_matrix,
            max_length=args.max_length,
            num_layers=args.transformer_layers,
            dropout=args.dropout,
            fine_tune_embeddings=args.fine_tune_embeddings,
        )
    raise ValueError(f"Unknown model: {spec.model_name}")


def run_training_job(
    spec: DnnExperimentSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DnnExperimentResult, list[EpochResult]]:
    start = time.perf_counter()
    model = model.to(device)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    epoch_results = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, y_true, y_pred = evaluate(model, val_loader, loss_fn, device)
        metrics = score_predictions(y_true, y_pred)
        epoch_result = EpochResult(
            experiment=spec.experiment,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            **metrics,
        )
        epoch_results.append(epoch_result)
        print(
            f"[epoch] {spec.experiment} | epoch: {epoch}/{args.epochs} "
            f"| val_score={epoch_result.cil_score:.5f} "
            f"| val_mae={epoch_result.mae:.5f}",
            flush=True,
        )

    best_epoch = max(epoch_results, key=lambda row: row.cil_score)
    train_seconds = time.perf_counter() - start
    result = DnnExperimentResult(
        experiment=spec.experiment,
        experiment_id=spec.experiment_id,
        embedding=spec.embedding_name,
        input_type=spec.input_type,
        representation=spec.representation,
        model=spec.model_name,
        report_rule=spec.report_rule,
        status="ok",
        best_epoch=best_epoch.epoch,
        train_seconds=train_seconds,
        accuracy=best_epoch.accuracy,
        macro_f1=best_epoch.macro_f1,
        mae=best_epoch.mae,
        cil_score=best_epoch.cil_score,
        quadratic_weighted_kappa=best_epoch.quadratic_weighted_kappa,
    )
    return result, epoch_results


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        optimizer.zero_grad()
        logits, labels = forward_batch(model, batch, device)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_examples += len(labels)
    return total_loss / max(1, total_examples)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch in loader:
            logits, labels = forward_batch(model, batch, device)
            loss = loss_fn(logits, labels)
            pred = logits.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            total_examples += len(labels)
            all_true.append(labels.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
    return (
        total_loss / max(1, total_examples),
        np.concatenate(all_true),
        np.concatenate(all_pred),
    )


def forward_batch(
    model: nn.Module,
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 2:
        vectors, labels = batch
        vectors = vectors.to(device)
        labels = labels.to(device)
        return model(vectors), labels
    token_ids, lengths, labels = batch
    token_ids = token_ids.to(device)
    lengths = lengths.to(device)
    labels = labels.to(device)
    return model(token_ids, lengths), labels


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[DnnExperimentResult], list[EpochResult]]:
    specs = experiment_specs(args)
    progress = ProgressTracker(total_jobs=len(specs))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    resources_by_embedding = {}
    for name, path in embedding_paths(args).items():
        print(f"Loading {name} embeddings from {path}")
        resources_by_embedding[name] = prepare_embedding_resources(
            name=name,
            path=path,
            train_texts=train_df["text_clean"],
        )
        resources = resources_by_embedding[name]
        print(
            f"{name} coverage: token={resources.token_coverage:.4f}, "
            f"vocab={resources.vocabulary_coverage:.4f}, dim={resources.dim}",
            flush=True,
        )

    labels_train = train_df["label"].to_numpy()
    labels_val = val_df["label"].to_numpy()
    sequence_cache = {}
    results = []
    epoch_history = []

    for spec in specs:
        progress.start(spec.experiment)
        try:
            resources = resources_by_embedding[spec.embedding_name]
            if spec.input_type == "document_vector":
                x_train = build_document_vectors(train_df["text_clean"], resources, spec.representation)
                x_val = build_document_vectors(val_df["text_clean"], resources, spec.representation)
                train_dataset = DocumentVectorDataset(x_train, labels_train)
                val_dataset = DocumentVectorDataset(x_val, labels_val)
                model = make_model(
                    spec=spec,
                    input_dim=x_train.shape[1],
                    embedding_matrix=None,
                    args=args,
                )
            else:
                if spec.embedding_name not in sequence_cache:
                    sequence_cache[spec.embedding_name] = build_sequence_inputs(
                        train_texts=train_df["text_clean"],
                        val_texts=val_df["text_clean"],
                        resources=resources,
                        max_length=args.max_length,
                        max_sequence_vocab=args.max_sequence_vocab,
                    )
                train_seq, train_len, val_seq, val_len, embedding_matrix = sequence_cache[spec.embedding_name]
                train_dataset = SequenceDataset(train_seq, train_len, labels_train)
                val_dataset = SequenceDataset(val_seq, val_len, labels_val)
                model = make_model(
                    spec=spec,
                    input_dim=None,
                    embedding_matrix=embedding_matrix,
                    args=args,
                )

            train_loader = make_loader(train_dataset, args.batch_size, shuffle=True)
            val_loader = make_loader(val_dataset, args.batch_size, shuffle=False)
            result, epochs = run_training_job(spec, train_loader, val_loader, model, args, device)
            results.append(result)
            epoch_history.extend(epochs)
            print_result(result)
        except Exception as exc:
            failed = DnnExperimentResult(
                experiment=spec.experiment,
                experiment_id=spec.experiment_id,
                embedding=spec.embedding_name,
                input_type=spec.input_type,
                representation=spec.representation,
                model=spec.model_name,
                report_rule=spec.report_rule,
                status="failed",
                notes=f"{type(exc).__name__}: {exc}",
            )
            results.append(failed)
            print_result(failed)

    return results, epoch_history


def print_result(result: DnnExperimentResult) -> None:
    if result.status != "ok":
        print(f"[failed] {result.experiment}: {result.notes}", flush=True)
        return
    print(
        f"[ok] {result.experiment}: "
        f"best_epoch={result.best_epoch} "
        f"score={result.cil_score:.5f} "
        f"mae={result.mae:.5f} "
        f"acc={result.accuracy:.5f} "
        f"macro_f1={result.macro_f1:.5f}",
        flush=True,
    )


def write_results(
    results: list[DnnExperimentResult],
    epoch_history: list[EpochResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "classic_dnn_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    history_path = output_dir / "classic_dnn_epoch_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        if epoch_history:
            writer = csv.DictWriter(f, fieldnames=list(asdict(epoch_history[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(row) for row in epoch_history)
        else:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "experiment",
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "accuracy",
                    "macro_f1",
                    "mae",
                    "cil_score",
                    "quadratic_weighted_kappa",
                ],
            )
            writer.writeheader()

    print(f"Wrote {results_path}")
    print(f"Wrote {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DNN baselines over static embeddings."
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/classic_dnn"))
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--glove-path", type=str, default="experiments/embeddings/glove.6B.300d.txt")
    parser.add_argument("--fasttext-path", type=str, default="experiments/embeddings/fasttext.vec")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--cnn-num-filters", type=int, default=128)
    parser.add_argument("--cnn-kernel-sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--include-transformers", action="store_true")
    parser.add_argument("--fine-tune-embeddings", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-sequence-vocab", type=int, default=200000)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    if args.glove_path and not Path(args.glove_path).exists():
        args.glove_path = None
    if args.fasttext_path and not Path(args.fasttext_path).exists():
        args.fasttext_path = None
    if not args.glove_path and not args.fasttext_path:
        raise FileNotFoundError(
            "No embedding files found. Run load_embeddings.ipynb first or pass "
            "--glove-path / --fasttext-path."
        )

    df = load_data(args.train_path)
    df["text_clean"] = df["sentence"].map(clean_text)
    if args.sample_size is not None and args.sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=args.sample_size,
            stratify=df["label"],
            random_state=args.random_state,
        )
        df = df.reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=args.validation_size,
        stratify=df["label"],
        random_state=args.random_state,
    )
    print(f"Train examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")

    results, epoch_history = run_experiments(train_df, val_df, args)
    write_results(results, epoch_history, args.output_dir)


if __name__ == "__main__":
    main()

