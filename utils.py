import logging

import numpy as np

logger = logging.getLogger(__name__)


def load_compute_metrics(pipeline_obj):
    """Return the appropriate compute_metrics callable for the pipeline's approach."""
    approach = getattr(pipeline_obj, "params", None)
    if approach is not None:
        approach = approach.approach
    if approach == "hpm":
        return _compute_metrics_hpm
    raise ValueError(
        f"No compute_metrics function registered for approach '{approach}'"
    )


def extract_model_logits(predictions) -> np.ndarray:
    """Extract raw logits from Trainer.predict() output, handling tuple wrapping."""
    raw = predictions.predictions
    if isinstance(raw, tuple):
        raw = raw[0]
    return raw


def _compute_metrics_hpm(eval_pred):
    logits, labels = eval_pred
    annotator_ids = logits[:, 0].astype(int)
    logits = logits[:, 1:]

    predictions = np.argmax(logits, axis=-1)
    metric_res = {}
    (
        metric_res["micro_accuracy"],
        metric_res["micro_precision"],
        metric_res["micro_recall"],
        metric_res["micro_f1"],
    ) = get_a_p_r_f(labels=labels, preds=predictions)
    all_f1s = []
    for annot_id in set(annotator_ids):
        annotator_labels = labels[annotator_ids == annot_id]
        annotator_preds = predictions[annotator_ids == annot_id]
        assert len(annotator_labels) == len(annotator_preds)
        _, _, _, f = get_a_p_r_f(
            labels=annotator_labels, preds=annotator_preds, agg_method="macro"
        )
        if len(annotator_labels) > 5:
            all_f1s.append(f)

    if all_f1s:
        metric_res["macro_f1"] = np.mean(all_f1s).round(3)
    else:
        # Fallback: use micro F1 when no annotator has > 5 examples
        logger.warning(
            "No annotator with > 5 examples found for macro F1; falling back to micro F1."
        )
        metric_res["macro_f1"] = metric_res["micro_f1"]
    return metric_res


def get_a_p_r_f(labels, preds, agg_method="macro"):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    a = accuracy_score(y_true=labels, y_pred=preds)
    p, r, f, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, average=agg_method, zero_division=0.0
    )

    scores = (
        round(a * 100, 3),
        round(p * 100, 3),
        round(r * 100, 3),
        round(f * 100, 3),
    )
    return scores
