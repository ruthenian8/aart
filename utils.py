import numpy as np


def load_compute_metrics(pipeline_obj):
    if "HPM" in str(type(pipeline_obj)):
        return _compute_metrics_hpm


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

    metric_res["macro_f1"] = np.mean(all_f1s).round(3)
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
