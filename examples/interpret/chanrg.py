from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import multimolecule  # noqa: F401  # Registers MultiMolecule auto models with transformers.
from multimolecule import AutoModelForSequencePrediction
from multimolecule.interpret import attribute, capture_activations, categorical_jacobian
from multimolecule.interpret.visualization import format_topk_substitutions


@dataclass
class ModelResult:
    model: str
    accuracy: float
    top_k_accuracy: float
    loss: float
    checkpoint_dir: str


class ChanrgDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        tokenizer,
        label_to_id: dict[str, int],
        *,
        max_length: int,
    ) -> None:
        self.table = table.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        encoded = self.tokenizer(
            row["sequence"],
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return {
            "id": row["id"],
            "sequence": row["sequence"],
            "label_name": row["label"],
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.label_to_id[row["label"]], dtype=torch.long),
        }


def collate_batch(batch: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    padded = tokenizer.pad(
        [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch],
        return_tensors="pt",
    )
    return {
        "ids": [item["id"] for item in batch],
        "sequences": [item["sequence"] for item in batch],
        "label_names": [item["label_name"] for item in batch],
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_table = load_table(args.train_path, args.target_column, args.max_train_samples, args.seed)
    eval_table = load_table(args.eval_path, args.target_column, args.max_eval_samples, args.seed)
    label_to_id = build_label_map(train_table)
    id_to_label = {index: label for label, index in label_to_id.items()}
    eval_table = eval_table[eval_table["label"].isin(label_to_id)].reset_index(drop=True)
    if eval_table.empty:
        raise ValueError("No evaluation examples remain after filtering labels unseen in the training split.")

    results: list[ModelResult] = []
    predictions_by_model: dict[str, pd.DataFrame] = {}
    for model_name in args.models:
        result, predictions = fine_tune_and_evaluate(
            model_name,
            train_table,
            eval_table,
            label_to_id,
            id_to_label,
            args,
            output_dir,
        )
        results.append(result)
        predictions_by_model[model_name] = predictions

    results = sorted(results, key=lambda result: (result.accuracy, result.top_k_accuracy, -result.loss), reverse=True)
    if len(results) < 2:
        raise ValueError("At least two models are required to compare best vs worst interpretability results.")
    best = results[0]
    worst = results[-1]

    metrics = {
        "task": "chanrg",
        "target_column": args.target_column,
        "num_labels": len(label_to_id),
        "metric_top_k": args.metric_top_k,
        "labels": id_to_label,
        "models": [asdict(result) for result in results],
        "best_model": best.model,
        "worst_model": worst.model,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    examples = select_interpretation_examples(
        predictions_by_model[best.model],
        predictions_by_model[worst.model],
        args.num_interpret_examples,
    )
    write_interpretations(
        examples,
        best,
        worst,
        label_to_id,
        args,
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune multiple MultiMolecule foundation models on CHANRG and run model-level "
            "interpretability on best vs worst models."
        )
    )
    parser.add_argument("--models", nargs="+", default=["multimolecule/calm", "multimolecule/rnafm"])
    parser.add_argument("--train-path", default="chanrg/train.parquet")
    parser.add_argument("--eval-path", default="chanrg/validation.parquet")
    parser.add_argument("--target-column", default="family")
    parser.add_argument("--output-dir", default="outputs/interpret/chanrg")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=2048)
    parser.add_argument("--max-eval-samples", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1016)
    parser.add_argument("--num-interpret-examples", type=int, default=8)
    parser.add_argument("--attribution-method", default="integrated_gradients")
    parser.add_argument("--attribution-steps", type=int, default=16)
    parser.add_argument("--jacobian-top-k", type=int, default=4)
    parser.add_argument("--jacobian-tokens", nargs="*", default=["A", "C", "G", "U"])
    parser.add_argument("--metric-top-k", type=int, default=5)
    parser.add_argument("--activation-layers", default="embeddings")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_table(path: str, target_column: str, max_samples: int | None, seed: int) -> pd.DataFrame:
    table = pd.read_parquet(path)
    required_columns = {"id", "sequence", target_column}
    missing_columns = sorted(required_columns.difference(table.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required CHANRG columns: {missing_columns}")
    table = table[["id", "sequence", target_column]].rename(columns={target_column: "label"})
    table = table.dropna(subset=["sequence", "label"]).copy()
    table["label"] = table["label"].astype(str)
    if max_samples is not None and max_samples > 0 and len(table) > max_samples:
        table = table.sample(n=max_samples, random_state=seed)
    return table.reset_index(drop=True)


def build_label_map(table: pd.DataFrame) -> dict[str, int]:
    labels = sorted(table["label"].unique().tolist())
    if len(labels) < 2:
        raise ValueError("CHANRG interpretation task requires at least two labels.")
    return {label: index for index, label in enumerate(labels)}


def fine_tune_and_evaluate(
    model_name: str,
    train_table: pd.DataFrame,
    eval_table: pd.DataFrame,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ModelResult, pd.DataFrame]:
    model_dir = output_dir / sanitize_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequencePrediction.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        head={"num_labels": len(label_to_id), "problem_type": "multiclass"},
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    ).to(args.device)

    train_loader = make_loader(train_table, tokenizer, label_to_id, args, shuffle=True)
    eval_loader = make_loader(eval_table, tokenizer, label_to_id, args, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model.train()
    for _ in range(args.num_epochs):
        for batch in tqdm(train_loader, desc=f"train {model_name}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            model_inputs = batch_to_model_inputs(batch, args.device)
            loss = model(**model_inputs).loss
            loss.backward()
            optimizer.step()

    eval_loss, predictions = evaluate(model, eval_loader, id_to_label, args.device, metric_top_k=args.metric_top_k)
    accuracy = float((predictions["predicted_label"] == predictions["label"]).mean())
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        ModelResult(
            model=model_name,
            accuracy=accuracy,
            top_k_accuracy=top_k_accuracy(predictions, k=args.metric_top_k),
            loss=eval_loss,
            checkpoint_dir=str(model_dir),
        ),
        predictions,
    )


def make_loader(
    table: pd.DataFrame,
    tokenizer,
    label_to_id: dict[str, int],
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = ChanrgDataset(table, tokenizer, label_to_id, max_length=args.max_length)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, tokenizer),
    )


def batch_to_model_inputs(batch: dict[str, Any], device: str) -> dict[str, Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    id_to_label: dict[int, str],
    device: str,
    *,
    metric_top_k: int,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    losses = []
    rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        model_inputs = batch_to_model_inputs(batch, device)
        outputs = model(**model_inputs)
        losses.append(float(outputs.loss.detach().cpu().item()))
        probabilities = outputs.logits.softmax(dim=-1).detach().cpu()
        predicted_ids = probabilities.argmax(dim=-1)
        top_k = min(probabilities.shape[-1], max(1, metric_top_k))
        top_k_ids = probabilities.topk(k=top_k, dim=-1).indices
        for row_index, predicted_id in enumerate(predicted_ids.tolist()):
            label_id = int(batch["labels"][row_index].item())
            rows.append(
                {
                    "id": batch["ids"][row_index],
                    "sequence": batch["sequences"][row_index],
                    "label": batch["label_names"][row_index],
                    "label_id": label_id,
                    "predicted_label": id_to_label[predicted_id],
                    "predicted_label_id": predicted_id,
                    "top_k_label_ids": top_k_ids[row_index].tolist(),
                    "confidence": float(probabilities[row_index, predicted_id].item()),
                    "correct_label_confidence": float(probabilities[row_index, label_id].item()),
                }
            )
    return float(sum(losses) / max(len(losses), 1)), pd.DataFrame(rows)


def top_k_accuracy(predictions: pd.DataFrame, *, k: int) -> float:
    if predictions.empty:
        return 0.0
    hits = [
        int(label_id in top_k_label_ids[:k])
        for label_id, top_k_label_ids in zip(predictions["label_id"], predictions["top_k_label_ids"])
    ]
    return float(sum(hits) / len(hits))


def select_interpretation_examples(
    best_predictions: pd.DataFrame,
    worst_predictions: pd.DataFrame,
    limit: int,
) -> pd.DataFrame:
    merged = best_predictions.merge(
        worst_predictions,
        on=["id", "sequence", "label", "label_id"],
        suffixes=("_best", "_worst"),
    )
    candidates = merged[
        (merged["predicted_label_best"] == merged["label"]) & (merged["predicted_label_worst"] != merged["label"])
    ]
    if len(candidates) == 0:
        candidates = merged.copy()
    candidates = candidates.sort_values(
        by=["correct_label_confidence_best", "correct_label_confidence_worst"],
        ascending=[False, True],
    )
    return candidates.head(limit).reset_index(drop=True)


def write_interpretations(
    examples: pd.DataFrame,
    best: ModelResult,
    worst: ModelResult,
    label_to_id: dict[str, int],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    path = output_dir / "interpretations.jsonl"
    with path.open("w") as handle:
        for role, result in (("best", best), ("worst", worst)):
            model_source = result.checkpoint_dir or result.model
            tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
            model = AutoModelForSequencePrediction.from_pretrained(
                model_source,
                num_labels=len(label_to_id),
                head={"num_labels": len(label_to_id), "problem_type": "multiclass"},
                label2id=label_to_id,
                id2label={index: label for label, index in label_to_id.items()},
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
            ).to(args.device)
            model.eval()
            for _, row in tqdm(examples.iterrows(), total=len(examples), desc=f"interpret {role}", leave=False):
                record = interpret_example(model, tokenizer, row, role, result.model, label_to_id, args)
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def interpret_example(
    model,
    tokenizer,
    row: pd.Series,
    role: str,
    model_name: str,
    label_to_id: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    encoded = tokenizer(
        row["sequence"],
        truncation=True,
        max_length=args.max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)
    target = label_to_id[row["label"]]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        probabilities = outputs.logits.softmax(dim=-1)[0].detach().cpu()
        predicted_id = int(probabilities.argmax().item())

    record: dict[str, Any] = {
        "model_role": role,
        "model": model_name,
        "id": row["id"],
        "label": row["label"],
        "target_label_id": target,
        "predicted_label_id": predicted_id,
        "predicted_confidence": float(probabilities[predicted_id].item()),
        "target_confidence": float(probabilities[target].item()),
        "tokens": tokens,
    }
    record.update(run_attribution(model, input_ids, attention_mask, target, args))
    record.update(run_jacobian(model, tokenizer, input_ids, attention_mask, target, args))
    record.update(run_activation_summary(model, input_ids, attention_mask, args))
    return record


def run_attribution(
    model,
    input_ids: Tensor,
    attention_mask: Tensor,
    target: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        output = attribute(
            model,
            input_ids,
            target=target,
            method=args.attribution_method,
            baseline="pad",
            attention_mask=attention_mask,
            n_steps=args.attribution_steps,
        )
        token_scores = output.token_attributions[0].detach().cpu()
        top_positions = token_scores.abs().topk(k=min(16, token_scores.numel())).indices.tolist()
        return {
            "attribution_method": output.method,
            "token_attributions": [float(value) for value in token_scores.tolist()],
            "top_attribution_positions": top_positions,
        }
    except Exception as error:  # pragma: no cover - example should continue when optional deps are absent.
        return {"attribution_error": str(error)}


def run_jacobian(
    model,
    tokenizer,
    input_ids: Tensor,
    attention_mask: Tensor,
    target: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        output = categorical_jacobian(
            model,
            input_ids,
            target=target,
            top_k=None if args.jacobian_tokens else args.jacobian_top_k,
            reduction="abs",
            attention_mask=attention_mask,
        )
        if args.jacobian_tokens:
            substitutions = format_canonical_substitutions(
                output,
                tokenizer,
                input_ids,
                candidate_tokens=args.jacobian_tokens,
                top_k=args.jacobian_top_k,
            )
        else:
            substitutions = format_topk_substitutions(output, vocabulary=get_index_to_token(tokenizer)).to_dict(
                orient="records"
            )
        return {"jacobian_topk_substitutions": substitutions}
    except Exception as error:  # pragma: no cover - example should continue when a model lacks embeddings.
        return {"jacobian_error": str(error)}


def run_activation_summary(model, input_ids: Tensor, attention_mask: Tensor, args: argparse.Namespace) -> dict[str, Any]:
    try:
        output = capture_activations(model, input_ids, attention_mask=attention_mask, layers=args.activation_layers)
        summary = {}
        for name, activation in output.activations.items():
            values = activation.detach().float().cpu()
            summary[name] = {
                "shape": list(values.shape),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
            }
        return {"activation_summary": summary}
    except Exception as error:  # pragma: no cover - layer names are intentionally user-configurable.
        return {"activation_error": str(error)}


def sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def get_index_to_token(tokenizer) -> list[str]:
    token_to_index = tokenizer.get_vocab()
    vocabulary = [""] * len(token_to_index)
    for token, index in token_to_index.items():
        if index < len(vocabulary):
            vocabulary[index] = token
    return vocabulary


def format_canonical_substitutions(
    output,
    tokenizer,
    input_ids: Tensor,
    *,
    candidate_tokens: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    candidate_ids = [tokenizer.convert_tokens_to_ids(token) for token in candidate_tokens]
    candidate_pairs = [
        (token, int(token_id))
        for token, token_id in zip(candidate_tokens, candidate_ids)
        if token_id is not None and token_id != tokenizer.unk_token_id
    ]
    if not candidate_pairs:
        raise ValueError(f"None of the requested Jacobian tokens are in the tokenizer vocabulary: {candidate_tokens}")

    positions = (
        output.positions.detach().cpu().tolist()
        if output.positions is not None
        else list(range(output.scores.shape[1]))
    )
    original_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    candidate_indices = torch.tensor([token_id for _, token_id in candidate_pairs], device=output.scores.device)
    scores = output.scores[0].index_select(dim=-1, index=candidate_indices).detach().cpu()
    top_k = min(top_k, scores.shape[-1])
    top_scores, top_indices = scores.topk(k=top_k, dim=-1)

    rows: list[dict[str, Any]] = []
    special_tokens = set(tokenizer.all_special_tokens)
    for row_index, position in enumerate(positions):
        original_token = original_tokens[position]
        if original_token in special_tokens:
            continue
        for rank in range(top_k):
            candidate_index = int(top_indices[row_index, rank].item())
            token, token_id = candidate_pairs[candidate_index]
            rows.append(
                {
                    "position": int(position),
                    "original_token": original_token,
                    "rank": rank + 1,
                    "token_index": token_id,
                    "token": token,
                    "score": float(top_scores[row_index, rank].item()),
                }
            )
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
