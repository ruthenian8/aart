import logging
import os

import numpy as np
import pandas as pd
import torch
from transformers import Trainer

from pipelines.generic_pipeline import GenericPipeline
from model_architectures import HyperLoRAModel
from utils import get_a_p_r_f, extract_model_logits

logger = logging.getLogger(__name__)


class HPMPipeline(GenericPipeline):
    def calculate_continous_disagreements(self, df, label_or_pred_col="label"):
        majority = df.groupby(self.instance_id_col)[label_or_pred_col].mean() >= 0.5
        count = df.groupby(self.instance_id_col)[label_or_pred_col].count()
        sum_ = df.groupby(self.instance_id_col)[label_or_pred_col].sum()

        disagreements = [
            (
                1.0 - float(sum_[t_i]) / float(count[t_i])
                if majority[t_i]
                else float(sum_[t_i]) / float(count[t_i])
            )
            for t_i in df[self.instance_id_col]
        ]
        return disagreements

    def calculate_majority(self, df, annotators=None):
        t = df.groupby(self.instance_id_col)["label"].agg(pd.Series.mode)
        aggregated_labels = t[df[self.instance_id_col]].tolist()
        # There could be a list of modes for each instance (in case there is more than one mode)
        # then we need to choose one of them
        aggregated_labels = [
            agg_vote if isinstance(agg_vote, (int, np.integer)) else agg_vote[0]
            for agg_vote in aggregated_labels
        ]
        aggregated_labels = [int(e) for e in aggregated_labels]
        assert "majority_label" not in df.columns
        return aggregated_labels

    def add_fake_annotators(self, df):
        N = self.params.num_fake_annotators
        if N <= 0:
            return df
        df_annotators = self.get_annotators(df)

        for i in range(N):
            for type_ in ["maj", "opp"]:
                fake_ann_name = f"annotator_fake_{type_}_{i}"
                assert fake_ann_name not in df_annotators
                logger.info("Adding %s", fake_ann_name)
                tmp_df = df.drop_duplicates(self.instance_id_col).copy()
                tmp_df = tmp_df.sample(
                    frac=1 / N, random_state=self.params.random_state
                )
                tmp_df["annotator"] = fake_ann_name
                tmp_df["label"] = np.abs(
                    tmp_df["majority_label"]
                    - np.random.choice([0, 1], size=tmp_df.shape[0], p=[0.9, 0.1])
                )
                if type_ == "opp":
                    tmp_df["label"] = 1 - tmp_df["label"]

                df = pd.concat([df, tmp_df], axis=0, ignore_index=True)

        return df.copy()

    def add_predictions(self, df, preds):
        logits = extract_model_logits(preds)
        # Strip the prepended annotator ID column
        class_logits = logits[:, 1:]
        df["pred"] = class_logits.argmax(axis=1)
        return df.copy()

    def encode_values(self, train_df, dev_df, test_df):
        encoding_colnames = self.params.embedding_colnames
        if not isinstance(encoding_colnames, list):
            encoding_colnames = []
        encoding_colnames = list(encoding_colnames)  # copy to avoid mutating params
        if "annotator" not in encoding_colnames:
            encoding_colnames = encoding_colnames + ["annotator"]
        logger.info("Encoding columns: %s", encoding_colnames)

        from sklearn.preprocessing import LabelEncoder

        label_encoders_dict = {}
        for emb_col in encoding_colnames:
            assert f"{emb_col}_int_encoded" not in train_df.columns
            label_encoders_dict[emb_col] = LabelEncoder()
            label_encoders_dict[emb_col].fit(train_df[emb_col].squeeze())
            self.data_dict[f"{emb_col}_map"] = {
                k: v
                for k, v in zip(
                    label_encoders_dict[emb_col].transform(
                        train_df[emb_col].squeeze().unique()
                    ),
                    train_df[emb_col].squeeze().unique(),
                )
            }

        for emb_col in encoding_colnames:
            train_df[f"{emb_col}_int_encoded"] = label_encoders_dict[emb_col].transform(
                train_df[emb_col].squeeze()
            )

            # Handle unknown annotators in dev/test based on policy
            unknown_policy = self.params.unknown_annotator_policy
            train_known_values = set(train_df[emb_col].unique())
            for split_name, split_df in [("dev", dev_df), ("test", test_df)]:
                unknown_mask = ~split_df[emb_col].isin(train_known_values)
                if unknown_mask.any():
                    n_unknown = unknown_mask.sum()
                    if unknown_policy == "error":
                        raise ValueError(
                            f"Found {n_unknown} rows in {split_name} with unknown {emb_col} values. "
                            f"Set unknown_annotator_policy='drop' to silently drop these rows."
                        )
                    elif unknown_policy == "drop":
                        logger.warning(
                            "Dropping %d rows from %s with unknown %s values",
                            n_unknown, split_name, emb_col,
                        )

            if unknown_policy == "drop":
                dev_df = (
                    dev_df[dev_df[emb_col].isin(train_known_values)]
                    .reset_index(drop=True)
                    .copy()
                )
                test_df = (
                    test_df[test_df[emb_col].isin(train_known_values)]
                    .reset_index(drop=True)
                    .copy()
                )

            dev_df[f"{emb_col}_int_encoded"] = label_encoders_dict[emb_col].transform(
                dev_df[emb_col].squeeze()
            )
            test_df[f"{emb_col}_int_encoded"] = label_encoders_dict[emb_col].transform(
                test_df[emb_col].squeeze()
            )

        return train_df, dev_df, test_df

    def _create_loss_label_weights(self, labels: pd.Series) -> dict:
        # Not used in the current HPM path; kept as abstract method implementation.
        return {}

    def _new_model(self, train_df):
        self.task_labels = None
        embd_type_cnt = {}
        for emb_col in ["annotator"]:
            embd_type_cnt[emb_col] = train_df[emb_col].nunique()
        logger.info("Embedding counts: %s", embd_type_cnt)

        train_labels_list = train_df.label.unique().astype(int).tolist()
        num_labels = len(set(train_labels_list))

        classifier = HyperLoRAModel.from_pretrained(
            pretrained_model_name_or_path=self.params.language_model_name,
            num_labels=num_labels,
            num_embeddings=embd_type_cnt["annotator"],
            device=self.device,
        )
        return classifier

    def get_batches(self, df):
        from datasets import Dataset

        ds_dict = {}
        ds_dict["labels"] = df["label"].values
        ds_dict["text_ids"] = df[self.instance_id_col].values
        ds_dict["text"] = df.prep_text.astype(str).tolist()
        ds_dict["annotator_ids"] = df["annotator_int_encoded"].values

        if "pair_id" in df.columns:
            ds_dict["parent_text"] = df.prep_parent_text.astype(str).tolist()

        ds = Dataset.from_dict(ds_dict)
        num_proc = self.params.num_tokenization_workers
        if num_proc is None:
            num_proc = min(os.cpu_count() or 1, 4)
        tokenized_ds = ds.map(
            self.tokenize_batch,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
        )
        tokenized_ds = tokenized_ds.remove_columns("text")
        if "parent_text" in tokenized_ds.column_names:
            tokenized_ds = tokenized_ds.remove_columns("parent_text")
        logger.info("Tokenized dataset: %s", tokenized_ds)
        return tokenized_ds

    def get_annotators(self, df):
        annotators_list = df.annotator.unique().tolist()
        return annotators_list

    def expand_test(self, df, unique_annotator_int):
        """Expand test set so every text is paired with every annotator."""
        cols_to_keep = [self.instance_id_col, "prep_text"]
        if self.instance_id_col == "pair_id" and "prep_parent_text" in df.columns:
            cols_to_keep.append("prep_parent_text")

        all_texts_df = df[cols_to_keep].drop_duplicates().copy()
        assert len(unique_annotator_int) == len(set(unique_annotator_int))
        all_annotators_df = pd.DataFrame(
            {"annotator_int_encoded": unique_annotator_int}
        )
        all_texts_all_annots = pd.merge(
            all_texts_df.assign(key=1), all_annotators_df.assign(key=1), on="key"
        ).drop("key", axis=1)
        all_texts_all_annots["label"] = np.nan

        logger.info("df shape before expanding: %s", df.shape)
        result_df = pd.concat([df, all_texts_all_annots], axis=0, ignore_index=True)
        result_df = result_df.drop_duplicates(
            [self.instance_id_col, "annotator_int_encoded"], keep="first"
        )
        logger.info("df shape after expanding: %s", result_df.shape)
        assert result_df.shape[0] == df[self.instance_id_col].nunique() * len(
            unique_annotator_int
        )
        return result_df

    def _calculate_majority_performance(self, trainer, test, train):
        import time

        unique_annotator_codes = test["annotator_int_encoded"].unique().tolist()

        start = time.time()
        test_expanded = self.expand_test(
            test, unique_annotator_int=unique_annotator_codes
        )
        end = time.time()
        logger.info("Time expanding test: %.2fs", end - start)

        # Track which rows have real labels vs synthetic (expanded) rows.
        # Synthetic rows have NaN labels and must NOT contribute to majority ground-truth.
        test_expanded["_has_real_label"] = test_expanded["label"].notna()

        # Fill NaN labels with a dummy value only for tokenization; they will not be
        # used for loss (labels column is removed before prediction) and are excluded
        # from the majority label computation below.
        test_expanded["label"] = test_expanded["label"].fillna(0).astype(int)

        test_dataset_expanded = self.get_batches(test_expanded)
        # Remove labels so model runs in inference-only mode
        test_dataset_expanded = test_dataset_expanded.remove_columns("labels")

        preds_expanded = trainer.predict(test_dataset_expanded)
        logits = extract_model_logits(preds_expanded)
        # Strip the prepended annotator ID column
        class_logits = logits[:, 1:]
        test_expanded["pred"] = class_logits.argmax(axis=1).tolist()
        assert test_expanded["pred"].isna().sum() == 0

        # Compute majority label using only the rows that had real ground-truth labels.
        # Using all rows (including synthetic ones filled with 0) would bias maj_label
        # toward 0 when annotators only labelled a subset of texts.
        real_labels_df = test_expanded[test_expanded["_has_real_label"]]
        maj_label_series = (
            real_labels_df.groupby(self.instance_id_col)["label"].mean() >= 0.5
        )

        # Majority prediction uses all annotators (that is the point of the expansion).
        test_expanded_results = test_expanded.groupby(self.instance_id_col)[
            ["pred"]
        ].mean()
        test_expanded_results["maj_label"] = maj_label_series
        test_expanded_results["maj_pred"] = test_expanded_results["pred"] >= 0.5
        scores_dict_test = {
            "type": "majority_all",
            "rand_seed": self.params.random_state,
        }
        (
            scores_dict_test["accuracy"],
            scores_dict_test["precision"],
            scores_dict_test["recall"],
            scores_dict_test["f1"],
        ) = get_a_p_r_f(
            labels=test_expanded_results["maj_label"],
            preds=test_expanded_results["maj_pred"],
        )
        return scores_dict_test, test_expanded

    def _calculate_annotator_performance(self, train, test):
        annotator_scores = []
        all_labels = []
        all_predictions = []
        all_f1s = []
        for a in train["annotator_int_encoded"].unique():
            a_results = test.loc[test["annotator_int_encoded"] == a].copy()
            if (
                a_results.empty
                or train.loc[train["annotator_int_encoded"] == a].shape[0] < 5
                or len(a_results) < 5
            ):
                continue
            logger.debug("Performance of %s", a_results["annotator"].iloc[0])
            scores_dict = {}
            (
                scores_dict["accuracy"],
                scores_dict["precision"],
                scores_dict["recall"],
                scores_dict["f1"],
            ) = get_a_p_r_f(labels=a_results["label"], preds=a_results["pred"])
            all_labels.append(a_results["label"])
            all_predictions.append(a_results["pred"])
            all_f1s.append(scores_dict["f1"])
            scores_dict["type"] = f"{a_results['annotator'].iloc[0]}"
            scores_dict["cnt_train"] = train.loc[
                train["annotator_int_encoded"] == a
            ].shape[0]
            scores_dict["cnt_test"] = a_results.shape[0]
            train_cnt_dict = (
                train.loc[train["annotator_int_encoded"] == a]["label"]
                .squeeze()
                .value_counts()
                .to_dict()
            )
            test_cnt_dict = a_results["label"].squeeze().value_counts().to_dict()

            scores_dict["cnt_contributions_train"] = train_cnt_dict
            scores_dict["cnt_contributions_test"] = test_cnt_dict
            scores_dict["cnt_train_positive"] = (
                train_cnt_dict[1] if 1 in train_cnt_dict else 0
            )
            scores_dict["cnt_test_positive"] = (
                test_cnt_dict[1] if 1 in test_cnt_dict else 0
            )
            scores_dict["sim_to_maj"] = round(
                (a_results["label"] == a_results["majority_label"]).mean(), 3
            )
            annotator_scores.append(scores_dict)

        all_labels = pd.concat(all_labels, axis=0, ignore_index=True)
        all_predictions = pd.concat(all_predictions, axis=0, ignore_index=True)
        scores_dict_test = {"type": "test", "rand_seed": self.params.random_state}
        (
            scores_dict_test["micro_accuracy"],
            scores_dict_test["micro_precision"],
            scores_dict_test["micro_recall"],
            scores_dict_test["micro_f1"],
        ) = get_a_p_r_f(labels=all_labels, preds=all_predictions)
        scores_dict_test["macro_f1"] = round(np.mean(all_f1s), 2) if all_f1s else 0.0
        annotator_scores.append(scores_dict_test)
        return annotator_scores
