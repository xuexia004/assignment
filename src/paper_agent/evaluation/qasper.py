"""QASPER official answer scoring, including all alternative annotations.

Protocol: allenai/qasper-led-baseline, scripts/evaluator.py. Gold annotations
belong only in evaluation; no generation or retrieval module imports this file.
"""

import re
import string
from collections import Counter


def token_f1(prediction: str, reference: str) -> float:
    def tokens(text):
        text = "".join(char for char in text.lower() if char not in string.punctuation)
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    predicted, gold = tokens(prediction), tokens(reference)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def reference_answers(annotations: list[dict]) -> list[dict]:
    """Preserve abstention references and the official answer-field precedence."""
    references = []
    for item in annotations:
        if item.get("unanswerable"):
            answer, kind = "Unanswerable", "none"
        elif item.get("extractive_spans"):
            answer, kind = ", ".join(item["extractive_spans"]), "extractive"
        elif item.get("free_form_answer"):
            answer, kind = item["free_form_answer"], "abstractive"
        elif item.get("yes_no"):
            answer, kind = "Yes", "boolean"
        elif item.get("yes_no") is not None:
            answer, kind = "No", "boolean"
        else:
            raise ValueError("QASPER annotation does not contain an answer")
        references.append({"answer": answer, "type": kind})
    if not references:
        raise ValueError("QASPER question does not contain annotations")
    return references


def score_answer(prediction: str, references: list[dict]) -> float:
    if not references:
        raise ValueError("An answer needs at least one reference")
    return max(token_f1(prediction, reference["answer"]) for reference in references)
