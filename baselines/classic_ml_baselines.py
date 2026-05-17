from __future__ import annotations

import argparse
import csv
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC


LABEL_MIN = 0
LABEL_MAX = 4
TOKEN_RE = re.compile(r"(?u)\b\w+\b")


@dataclass(frozen=True)
class ExperimentResult:
    experiment: str
    family: str
    representation: str
    variant: str
    classifier: str
    status: str
    train_seconds: float | None = None
    predict_seconds: float | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    mae: float | None = None
    cil_score: float | None = None
    quadratic_weighted_kappa: float | None = None
    notes: str = ""


@dataclass
class ProgressTracker:
    total_jobs: int
    jobs_run: int = 0

    def start(self, job_name: str) -> None:
        self.jobs_run += 1
        print(
            f"[job] {job_name} | jobs_run: {self.jobs_run}/{self.total_jobs}",
            flush=True,
        )


def clean_text(text: object) -> str:
    """Normalize whitespace while preserving punctuation, casing, and emojis."""
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize_words(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_data(train_path: Path) -> pd.DataFrame:
    return pd.read_csv(train_path)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=int)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mae": mae,
        "cil_score": 1.0 - mae / 4.0,
        "quadratic_weighted_kappa": cohen_kappa_score(
            y_true,
            y_pred,
            labels=list(range(LABEL_MIN, LABEL_MAX + 1)),
            weights="quadratic",
        ),
    }


def make_dense_classifiers(max_iter: int) -> dict[str, BaseEstimator]:
    return {
        "logreg": LogisticRegression(C=1.0, max_iter=max_iter),
        #"linear_svm": LinearSVC(C=1.0),
        #"ridge_classifier": RidgeClassifier(alpha=1.0),
    }


def make_sparse_classifiers(max_iter: int) -> dict[str, BaseEstimator]:
    classifiers = make_dense_classifiers(max_iter=max_iter)
    classifiers["complement_nb"] = ComplementNB(alpha=1.0)
    return classifiers


def sparse_feature_specs(max_features: int) -> list[tuple[str, str, str, BaseEstimator]]:
    return [
        (
            "bow",
            "word_1_1",
            "word n-grams=(1, 1)",
            CountVectorizer(
                preprocessor=clean_text,
                lowercase=True,
                ngram_range=(1, 1),
                max_features=max_features,
                min_df=2,
            ),
        ),
        (
            "bow",
            "word_1_2",
            "word n-grams=(1, 2)",
            CountVectorizer(
                preprocessor=clean_text,
                lowercase=True,
                ngram_range=(1, 2),
                max_features=max_features,
                min_df=2,
            ),
        ),
        (
            "bow",
            "word_1_3",
            "word n-grams=(1, 3)",
            CountVectorizer(
                preprocessor=clean_text,
                lowercase=True,
                ngram_range=(1, 3),
                max_features=max_features,
                min_df=2,
            ),
        ),
        (
            "tfidf_word",
            "word_1_1",
            "word n-grams=(1, 1)",
            TfidfVectorizer(
                preprocessor=clean_text,
                lowercase=True,
                ngram_range=(1, 1),
                max_features=max_features,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "tfidf_word",
            "word_1_2",
            "word n-grams=(1, 2)",
            TfidfVectorizer(
                preprocessor=clean_text,
                lowercase=True,
                ngram_range=(1, 2),
                max_features=max_features,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "tfidf_word",
            "word_1_3",
            "word n-grams=(1, 3)",
            TfidfVectorizer(
                preprocessor=clean_text,
                lowercase=True,
                ngram_range=(1, 3),
                max_features=max_features,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "tfidf_char",
            "char_wb_3_5",
            "char_wb n-grams=(3, 5)",
            TfidfVectorizer(
                preprocessor=clean_text,
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=max_features,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "tfidf_char",
            "char_wb_3_6",
            "char_wb n-grams=(3, 6)",
            TfidfVectorizer(
                preprocessor=clean_text,
                lowercase=True,
                analyzer="char_wb",
                ngram_range=(3, 6),
                max_features=max_features,
                min_df=2,
                sublinear_tf=True,
            ),
        ),
    ]


class StaticEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Turn a document into one vector using pretrained static word vectors."""

    def __init__(
        self,
        embedding_path: str,
        strategy: str = "average",
        sif_a: float = 1e-3,
        max_sif_documents: int = 50000,
    ):
        self.embedding_path = embedding_path
        self.strategy = strategy
        self.sif_a = sif_a
        self.max_sif_documents = max_sif_documents

    def fit(self, texts: Iterable[str], y=None):
        texts = list(texts)
        tokenized = [tokenize_words(clean_text(text)) for text in texts]
        counts = Counter(token for tokens in tokenized for token in tokens)
        self.total_tokens_ = sum(counts.values())
        self.term_counts_ = counts
        self.idf_ = self._compute_idf(tokenized)
        self.embeddings_, self.dim_, self.coverage_ = self._load_embeddings(set(counts))
        if self.strategy == "sif":
            sif_vectors = self._documents_to_matrix(
                tokenized[: self.max_sif_documents],
                remove_sif_pc=False,
            )
            self.sif_pc_ = self._first_pc(sif_vectors)
        else:
            self.sif_pc_ = None
        return self

    def transform(self, texts: Iterable[str]):
        tokenized = [tokenize_words(clean_text(text)) for text in texts]
        if self.strategy == "sif":
            return self._documents_to_matrix(tokenized, remove_sif_pc=True)
        return self._documents_to_matrix(tokenized, remove_sif_pc=False)

    def _compute_idf(self, tokenized: list[list[str]]) -> dict[str, float]:
        n_docs = len(tokenized)
        df = Counter()
        for tokens in tokenized:
            df.update(set(tokens))
        return {
            token: math.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
            for token, doc_freq in df.items()
        }

    def _load_embeddings(
        self, vocabulary: set[str]
    ) -> tuple[dict[str, np.ndarray], int, dict[str, float]]:
        path = Path(self.embedding_path)
        if not path.exists():
            raise FileNotFoundError(path)

        embeddings: dict[str, np.ndarray] = {}
        dim: int | None = None
        seen_lines = 0

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
            maybe_header = first.strip().split()
            if len(maybe_header) == 2 and all(part.isdigit() for part in maybe_header):
                pass
            else:
                self._try_add_embedding_line(first, vocabulary, embeddings)
                if embeddings:
                    dim = len(next(iter(embeddings.values())))

            for line in f:
                seen_lines += 1
                if self._try_add_embedding_line(line, vocabulary, embeddings):
                    if dim is None:
                        dim = len(next(iter(embeddings.values())))
                if len(embeddings) == len(vocabulary):
                    break

        if dim is None and embeddings:
            dim = len(next(iter(embeddings.values())))
        if dim is None:
            raise ValueError(f"No usable vectors found in {path}")

        covered_token_count = sum(self.term_counts_[token] for token in embeddings)
        coverage = {
            "vocabulary_coverage": len(embeddings) / max(1, len(vocabulary)),
            "token_coverage": covered_token_count / max(1, self.total_tokens_),
            "loaded_vectors": float(len(embeddings)),
            "scanned_lines": float(seen_lines),
        }
        return embeddings, dim, coverage

    @staticmethod
    def _try_add_embedding_line(
        line: str,
        vocabulary: set[str],
        embeddings: dict[str, np.ndarray],
    ) -> bool:
        parts = line.rstrip().split(" ")
        if len(parts) <= 2:
            return False
        token = parts[0].lower()
        if token not in vocabulary:
            return False
        vector = np.fromstring(" ".join(parts[1:]), sep=" ", dtype=np.float32)
        if vector.size == 0:
            return False
        embeddings[token] = vector
        return True

    def _documents_to_matrix(
        self,
        tokenized: list[list[str]],
        remove_sif_pc: bool,
    ) -> np.ndarray:
        rows = [self._document_vector(tokens) for tokens in tokenized]
        matrix = np.vstack(rows).astype(np.float32)
        if remove_sif_pc and self.sif_pc_ is not None:
            pc = self.sif_pc_.reshape(1, -1)
            matrix = matrix - matrix.dot(pc.T) * pc
        return matrix

    def _document_vector(self, tokens: list[str]) -> np.ndarray:
        vectors = []
        weights = []
        for token in tokens:
            vector = self.embeddings_.get(token)
            if vector is None:
                continue
            vectors.append(vector)
            weights.append(self._token_weight(token))

        if not vectors:
            if self.strategy == "mean_max":
                return np.zeros(self.dim_ * 2, dtype=np.float32)
            return np.zeros(self.dim_, dtype=np.float32)

        mat = np.vstack(vectors).astype(np.float32)
        if self.strategy == "mean_max":
            return np.concatenate([mat.mean(axis=0), mat.max(axis=0)])

        weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        return (mat * weights_arr).sum(axis=0) / max(float(weights_arr.sum()), 1e-12)

    def _token_weight(self, token: str) -> float:
        if self.strategy == "tfidf_weighted":
            return self.idf_.get(token, 1.0)
        if self.strategy == "sif":
            probability = self.term_counts_.get(token, 0) / max(1, self.total_tokens_)
            return self.sif_a / (self.sif_a + probability)
        return 1.0

    @staticmethod
    def _first_pc(matrix: np.ndarray) -> np.ndarray | None:
        nonzero = matrix[np.linalg.norm(matrix, axis=1) > 0]
        if len(nonzero) < 2:
            return None
        centered = nonzero - nonzero.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return vt[0].astype(np.float32)


def embedding_feature_specs(args: argparse.Namespace) -> list[tuple[str, str, str, BaseEstimator]]:
    specs = []
    paths = {
        "glove": args.glove_path,
        "fasttext": args.fasttext_path,
        "word2vec": args.word2vec_path,
    }
    strategies = ["average", "tfidf_weighted", "mean_max", "sif"]
    for family, path in paths.items():
        if not path:
            continue
        for strategy in strategies:
            specs.append(
                (
                    family,
                    strategy,
                    f"{family} {strategy}",
                    Pipeline(
                        [
                            (
                                "embedding",
                                StaticEmbeddingVectorizer(
                                    embedding_path=path,
                                    strategy=strategy,
                                    max_sif_documents=args.max_sif_documents,
                                ),
                            ),
                            ("normalize", Normalizer(norm="l2")),
                        ]
                    ),
                )
            )
    return specs


def run_estimator(
    experiment: str,
    family: str,
    representation: str,
    variant: str,
    classifier_name: str,
    estimator: BaseEstimator,
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    notes: str = "",
    progress: ProgressTracker | None = None,
) -> ExperimentResult:
    if progress is not None:
        progress.start(experiment)
    try:
        start = time.perf_counter()
        estimator.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        pred = estimator.predict(x_val)
        predict_seconds = time.perf_counter() - start
        metrics = score_predictions(y_val, pred)
        return ExperimentResult(
            experiment=experiment,
            family=family,
            representation=representation,
            variant=variant,
            classifier=classifier_name,
            status="ok",
            train_seconds=train_seconds,
            predict_seconds=predict_seconds,
            notes=notes,
            **metrics,
        )
    except Exception as exc:
        return ExperimentResult(
            experiment=experiment,
            family=family,
            representation=representation,
            variant=variant,
            classifier=classifier_name,
            status="failed",
            notes=f"{type(exc).__name__}: {exc}",
        )


def run_dummy_baseline(
    x_train: pd.Series,
    y_train: np.ndarray,
    x_val: pd.Series,
    y_val: np.ndarray,
    progress: ProgressTracker,
) -> ExperimentResult:
    return run_estimator(
        experiment="majority__most_frequent",
        family="majority",
        representation="constant",
        variant="most_frequent",
        classifier_name="dummy_most_frequent",
        estimator=DummyClassifier(strategy="most_frequent"),
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        notes="Always predicts the most frequent training label.",
        progress=progress,
    )


def run_sparse_experiments(
    x_train: pd.Series,
    y_train: np.ndarray,
    x_val: pd.Series,
    y_val: np.ndarray,
    max_features: int,
    max_iter: int,
    progress: ProgressTracker,
) -> list[ExperimentResult]:
    results = []
    classifiers = make_sparse_classifiers(max_iter=max_iter)
    specs = sparse_feature_specs(max_features)

    for family, variant, description, vectorizer in specs:
        for classifier_name, classifier in classifiers.items():
            experiment = f"{family}__{variant}__{classifier_name}"
            estimator = Pipeline(
                [("features", clone(vectorizer)), ("classifier", clone(classifier))]
            )
            results.append(
                run_estimator(
                    experiment=experiment,
                    family=family,
                    representation=description,
                    variant=variant,
                    classifier_name=classifier_name,
                    estimator=estimator,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    progress=progress,
                )
            )
            print_result(results[-1])
    return results


def run_embedding_experiments(
    x_train: pd.Series,
    y_train: np.ndarray,
    x_val: pd.Series,
    y_val: np.ndarray,
    args: argparse.Namespace,
    progress: ProgressTracker,
) -> list[ExperimentResult]:
    results = []
    classifiers = make_dense_classifiers(max_iter=args.max_iter)
    specs = embedding_feature_specs(args)

    for family, variant, description, transformer in specs:
        feature_extraction_job = f"{family}__{variant}__features"
        progress.start(feature_extraction_job)
        try:
            start = time.perf_counter()
            x_train_dense = transformer.fit_transform(x_train)
            x_val_dense = transformer.transform(x_val)
            feature_seconds = time.perf_counter() - start
            coverage = transformer.named_steps["embedding"].coverage_
            notes = (
                f"feature_seconds={feature_seconds:.2f}; "
                f"vocab_cov={coverage['vocabulary_coverage']:.4f}; "
                f"token_cov={coverage['token_coverage']:.4f}; "
                f"loaded_vectors={int(coverage['loaded_vectors'])}"
            )
        except Exception as exc:
            failed = ExperimentResult(
                experiment=feature_extraction_job,
                family=family,
                representation=description,
                variant=variant,
                classifier="feature_extraction",
                status="failed",
                notes=f"{type(exc).__name__}: {exc}",
            )
            results.append(failed)
            print_result(failed)
            continue

        for classifier_name, classifier in classifiers.items():
            experiment = f"{family}__{variant}__{classifier_name}"
            results.append(
                run_estimator(
                    experiment=experiment,
                    family=family,
                    representation=description,
                    variant=variant,
                    classifier_name=classifier_name,
                    estimator=clone(classifier),
                    x_train=x_train_dense,
                    y_train=y_train,
                    x_val=x_val_dense,
                    y_val=y_val,
                    notes=notes,
                    progress=progress,
                )
            )
            print_result(results[-1])
    return results


def print_result(result: ExperimentResult) -> None:
    if result.status != "ok":
        print(f"[failed] {result.experiment}: {result.notes}")
        return
    print(
        f"[ok] {result.experiment}: "
        f"score={result.cil_score:.5f} "
        f"mae={result.mae:.5f} "
        f"acc={result.accuracy:.5f} "
        f"macro_f1={result.macro_f1:.5f}"
    )


def count_jobs(args: argparse.Namespace) -> int:
    total = 1
    if args.mode in {"sparse", "all"}:
        total += len(sparse_feature_specs(args.max_features)) * len(
            make_sparse_classifiers(max_iter=args.max_iter)
        )
    if args.mode in {"embeddings", "all"}:
        total += len(embedding_feature_specs(args)) * (
            1 + len(make_dense_classifiers(max_iter=args.max_iter))
        )
    return total


def write_results(results: list[ExperimentResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    results_path = output_dir / "classic_ml_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    best_by_family = {}
    for row in ok_rows:
        current = best_by_family.get(row["family"])
        if current is None or row["cil_score"] > current["cil_score"]:
            best_by_family[row["family"]] = row

    best_path = output_dir / "classic_ml_best_by_family.csv"
    with best_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(best_by_family.values())

    print(f"Wrote {results_path}")
    print(f"Wrote {best_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classic ML baselines for the CIL sentiment project."
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/classic_ml"))
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["sparse", "embeddings", "all"],
        default="all",
        help="Choose which baseline families to evaluate.",
    )
    parser.add_argument("--glove-path", type=str, default=None)
    parser.add_argument("--fasttext-path", type=str, default=None)
    parser.add_argument("--word2vec-path", type=str, default=None)
    parser.add_argument("--max-sif-documents", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    x_train = train_df["text_clean"]
    y_train = train_df["label"].to_numpy()
    x_val = val_df["text_clean"]
    y_val = val_df["label"].to_numpy()

    print(f"Train examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    class_counts = {int(label): int(count) for label, count in sorted(Counter(y_train).items())}
    print(f"Class counts: {class_counts}")

    progress = ProgressTracker(total_jobs=count_jobs(args))

    results = [run_dummy_baseline(x_train, y_train, x_val, y_val, progress)]
    print_result(results[-1])

    if args.mode in {"sparse", "all"}:
        results.extend(
            run_sparse_experiments(
                x_train,
                y_train,
                x_val,
                y_val,
                max_features=args.max_features,
                max_iter=args.max_iter,
                progress=progress,
            )
        )
    if args.mode in {"embeddings", "all"}:
        results.extend(
            run_embedding_experiments(
                x_train,
                y_train,
                x_val,
                y_val,
                args=args,
                progress=progress,
            )
        )

    write_results(results, args.output_dir)


if __name__ == "__main__":
    main()
