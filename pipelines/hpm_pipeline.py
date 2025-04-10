import torch
import numpy as np
import pandas as pd
from transformers import Trainer
from pipelines.generic_pipeline import GenericPipeline
from model_architectures import HyperPeftModel
from sklearn.utils.class_weight import compute_class_weight
from utils import get_a_p_r_f


class HPMPipeline(GenericPipeline):
    def calculate_continous_disagreements(self, df, label_or_pred_col="label"):
        majority = df.groupby(self.instance_id_col)[label_or_pred_col].mean() >= 0.5
        count = df.groupby(self.instance_id_col)[label_or_pred_col].count()
        sum = df.groupby(self.instance_id_col)[label_or_pred_col].sum()

        disagreements = [
            (
                1.0 - float(sum[t_i]) / float(count[t_i])
                if majority[t_i]
                else float(sum[t_i]) / float(count[t_i])
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

    def _create_loss_label_weights(self, labels):
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(labels), y=labels
        )
        print("Weights used for labels: ", weights)
        if len(weights) == 1:
            weights = [0.01, 1]
        weights = torch.tensor(
            weights, dtype=torch.bfloat16, device="cuda"
        )  # .to(self.device)
        return weights

    def _create_loss_annotator_weights(self, annotators):
        annot_codes = np.unique(annotators)
        for i in range(len(annot_codes)):
            assert annot_codes[i] == i
        weights = compute_class_weight(
            class_weight="balanced", classes=annot_codes, y=annotators
        )
        # print("Weights used for annotators: ", weights)
        if len(weights) == 1:
            weights = [0.01, 1]
        weights = torch.tensor(
            weights, dtype=torch.bfloat16, device="cuda"
        )  # .to(self.device)
        return weights

    def add_fake_annotators(self, df):
        N = self.params.num_fake_annotators
        if N <= 0:
            return df
        df_annotators = self.get_annotators(df)

        for i in range(N):
            for type in ["maj", "opp"]:
                fake_ann_name = f"annotator_fake_{type}_{i}"
                assert fake_ann_name not in df_annotators
                print(f"*** Adding {fake_ann_name}")
                tmp_df = df.drop_duplicates(self.instance_id_col).copy()
                tmp_df = tmp_df.sample(
                    frac=1 / N, random_state=self.params.random_state
                )
                tmp_df["annotator"] = fake_ann_name
                # todo this should change for multi class
                tmp_df["label"] = np.abs(
                    tmp_df["majority_label"]
                    - np.random.choice([0, 1], size=tmp_df.shape[0], p=[0.9, 0.1])
                )
                if type == "opp":
                    tmp_df["label"] = 1 - tmp_df["label"]

                df = pd.concat([df, tmp_df], axis=0, ignore_index=True)

        return df.copy()

    def add_predictions(self, df, preds):
        df["pred"] = preds.predictions[:, 1:].argmax(axis=1)
        return df.copy()

    def encode_values(self, train_df, dev_df, test_df):
        encoding_colnames = self.params.embedding_colnames
        if "annotator" not in encoding_colnames:
            encoding_colnames = encoding_colnames + ["annotator"]
        print("encoding colnames: ")
        print(encoding_colnames)

        from sklearn.preprocessing import LabelEncoder

        label_encoders_dict = {}
        ### integer mapping using LabelEncoder
        print("Encoding the following columns: ", encoding_colnames)
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

        # TODO remove the following from the main branch and only keep in emfd branch
        for emb_col in encoding_colnames:
            ignore_error = False
            if (emb_col == "annotator") and ("emfd" in self.params.data_name.lower()):
                ignore_error = True
            train_df[f"{emb_col}_int_encoded"] = label_encoders_dict[emb_col].transform(
                train_df[emb_col].squeeze()
            )
            if ignore_error:
                print(dev_df.shape)
                dev_df = (
                    dev_df[dev_df[emb_col].isin(train_df[emb_col])]
                    .reset_index(drop=True)
                    .copy()
                )
                print(dev_df.shape)
                print(test_df.shape)
                test_df = (
                    test_df[test_df[emb_col].isin(train_df[emb_col])]
                    .reset_index(drop=True)
                    .copy()
                )
                print(test_df.shape)

            dev_df[f"{emb_col}_int_encoded"] = label_encoders_dict[emb_col].transform(
                dev_df[emb_col].squeeze()
            )
            test_df[f"{emb_col}_int_encoded"] = label_encoders_dict[emb_col].transform(
                test_df[emb_col].squeeze()
            )

        return train_df, dev_df, test_df

    def _new_model(self, train_df):
        self.task_labels = None
        embd_type_cnt = {}
        for emb_col in ["annotator"]:
            embd_type_cnt[emb_col] = train_df[emb_col].nunique()
        print(embd_type_cnt)

        train_labels_list = train_df.label.unique().astype(int).tolist()
        num_labels = len(set(train_labels_list))

        classifier = HyperPeftModel.from_pretrained(
            pretrained_model_name_or_path=self.params.language_model_name,
            num_labels=num_labels,
            num_embeddings=embd_type_cnt["annotator"],
            embedding_dim=768,
            layer_embedding_dim=256,
        )
        return classifier

    def get_batches(self, df):
        from datasets import Dataset

        ds_dict = {}
        ds_dict["labels"] = df["label"].values
        ds_dict["text_ids"] = df[self.instance_id_col].values
        ds_dict["text"] = df.prep_text.astype(str)
        ds_dict[f"annotator_ids"] = df[f"annotator_int_encoded"]

        if "pair_id" in df:
            ds_dict["parent_text"] = df.prep_parent_text.astype(str)
        ds = Dataset.from_dict(ds_dict)
        tokenized_ds = ds.map(lambda x: self.tokenize_function(x), num_proc=16)
        tokenized_ds = tokenized_ds.remove_columns("text")
        if "pair_id" in df:
            tokenized_ds = tokenized_ds.remove_columns("parent_text")
        print(tokenized_ds)
        return tokenized_ds

    def get_annotators(self, df):
        annotators_list = df.annotator.unique().tolist()
        return annotators_list

    def expand_test(self, df, unique_annotator_int):
        all_texts_df = df[[self.instance_id_col, "prep_text"]].drop_duplicates().copy()
        assert len(unique_annotator_int) == len((set(unique_annotator_int)))
        all_annotators_df = pd.DataFrame(
            {"annotator_int_encoded": unique_annotator_int}
        )
        all_texts_all_annots = pd.merge(
            all_texts_df.assign(key=1), all_annotators_df.assign(key=1), on="key"
        ).drop("key", axis=1)
        all_texts_all_annots["label"] = np.nan

        print("df shape before appending the missing annotators: ", df.shape)
        result_df = pd.concat([df, all_texts_all_annots], axis=0, ignore_index=True)
        result_df = result_df.drop_duplicates(
            [self.instance_id_col, "annotator_int_encoded"], keep="first"
        )
        print("df shape after appending the missing annotators: ", result_df.shape)
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
        print(f"Time spent on expanding test {end - start}")
        test_dataset_expanded = self.get_batches(test_expanded)
        test_dataset_expanded = test_dataset_expanded.remove_columns("labels")
        preds_expanded = trainer.predict(test_dataset_expanded)
        test_expanded["pred"] = (
            preds_expanded.predictions[0][:, 1:].argmax(axis=1).tolist()
        )
        assert test_expanded["pred"].isna().sum() == 0

        test_expanded_results = test_expanded.groupby(self.instance_id_col)[
            ["label", "pred"]
        ].mean()
        test_expanded_results["maj_label"] = test_expanded_results["label"] >= 0.5
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
            print("~" * 30)
            print("~" * 30)
            print(f" * * * Performance of {a_results['annotator'].iloc[0]}")
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
            print(scores_dict)
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
        scores_dict_test["macro_f1"] = round(np.mean(all_f1s), 2)
        annotator_scores.append(scores_dict_test)
        return annotator_scores
