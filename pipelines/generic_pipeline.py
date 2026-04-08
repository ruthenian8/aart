import abc
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback

from params import Params
from utils import load_compute_metrics

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed, deterministic=True):
    import random
    import transformers

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


@dataclass
class MyTrainingArguments(TrainingArguments):
    lambda2: float = field(
        default=0.0,
        metadata={
            "help": "lambda2 coefficient for l2_norm of annotator embeddings in the loss"
        },
    )
    contrastive_alpha: float = field(
        default=0.0,
        metadata={
            "help": "The coefficient for contrastive loss computed for annotator embeddings"
        },
    )
    shuffle_train_data: bool = field(
        default=True, metadata={"help": "Whether to shuffle input data"}
    )


class GenericPipeline(abc.ABC):
    """Abstract base class for classifier pipelines."""

    def __init__(self, params: Params):
        # Resolve device
        if params.device:
            self.device = torch.device(params.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info("Device: %s", self.device)
        if self.device.type == "cuda":
            gpu_id = torch.cuda.current_device()
            logger.info(
                "CUDA device %d: %s", gpu_id, torch.cuda.get_device_name(gpu_id)
            )
            logger.info("GPU count: %d", torch.cuda.device_count())

        set_seed(params.random_state, deterministic=params.deterministic)
        self.params = params
        logger.info("Params: %s", self.params)
        self.data_dict = self.read_data()

        self.tokenizer = AutoTokenizer.from_pretrained(self.params.language_model_name)

        self.compute_metrics_function = load_compute_metrics(self)
        logger.info(
            "Pipeline initialized: %s",
            [(k, v) for k, v in self.params.__dict__.items()],
        )

    def get_param_combinations(self, sep1="_", sep2=", ", exclude_list=None):
        if exclude_list is None:
            exclude_list = []
        exclude_list = exclude_list + [
            "skip_test",
            "skip_majority",
            "use_majority_weight",
            "num_epochs",
            "early_stopping_patience",
            "device",
            "num_tokenization_workers",
            "deterministic",
            "unknown_annotator_policy",
        ]
        param_combinations = sep2.join(
            key + sep1 + str(val)
            for key, val in self.params.__dict__.items()
            if key not in exclude_list
        )

        return param_combinations

    def read_data(self):
        logger.info("Reading data: %s", self.params.data_name)
        data_name_for_path = self.params.data_name
        data_path = REPO_ROOT / "data" / self.params.approach / data_name_for_path

        data_dict = {}

        if isinstance(self.params.embedding_colnames, str):
            if self.params.embedding_colnames.strip() == "":
                self.params.embedding_colnames = []
            else:
                self.params.embedding_colnames = self.params.embedding_colnames.split(
                    ","
                )

        df = pd.read_csv(data_path / "all_data.csv")
        if self.params.embedding_colnames:
            df[self.params.embedding_colnames] = df[
                self.params.embedding_colnames
            ].replace(["Do Not Wish to Answer", "MISSING"], "unknown")
            df[self.params.embedding_colnames] = df[
                self.params.embedding_colnames
            ].fillna("unknown")

        if "label" in df.columns:
            df["label"] = df["label"].astype(int)

        self.instance_id_col = "pair_id" if "pair_id" in df.columns else "text_id"
        logger.info("Instance id column: %s", self.instance_id_col)
        df_annotators = self.get_annotators(df)
        df["majority_label"] = self.calculate_majority(df, annotators=df_annotators)
        if "disagreement_level" not in df.columns:
            df["disagreement_level"] = self.calculate_continous_disagreements(
                df=df, label_or_pred_col="label"
            )
            df["disagreement_level"] = df["disagreement_level"].round(1)

        df = self.add_fake_annotators(df)

        splits_path = REPO_ROOT / "splits" / data_name_for_path
        train_idx = (
            (splits_path / f"train_{self.params.random_state}.txt")
            .read_text()
            .splitlines()
        )
        dev_idx = (
            (splits_path / f"dev_{self.params.random_state}.txt")
            .read_text()
            .splitlines()
        )
        test_idx = (
            (splits_path / f"test_{self.params.random_state}.txt")
            .read_text()
            .splitlines()
        )

        assert set(df[self.instance_id_col].astype(str)) == set(
            train_idx + dev_idx + test_idx
        )

        data_dict["train"] = df.loc[
            df[self.instance_id_col].astype(str).isin(train_idx)
        ].reset_index(drop=True)
        data_dict["dev"] = df.loc[
            df[self.instance_id_col].astype(str).isin(dev_idx)
        ].reset_index(drop=True)
        data_dict["test"] = df.loc[
            df[self.instance_id_col].astype(str).isin(test_idx)
        ].reset_index(drop=True)

        if self.params.sort_instances_by:
            data_dict["train"] = data_dict["train"].sort_values(
                by=[self.params.sort_instances_by]
            )

        assert set(data_dict["train"][self.instance_id_col]).isdisjoint(
            set(data_dict["test"][self.instance_id_col])
        )
        assert set(data_dict["train"][self.instance_id_col]).isdisjoint(
            set(data_dict["dev"][self.instance_id_col])
        )
        assert set(data_dict["dev"][self.instance_id_col]).isdisjoint(
            set(data_dict["test"][self.instance_id_col])
        )
        assert not (
            data_dict["train"].empty
            or data_dict["dev"].empty
            or data_dict["test"].empty
        )

        logger.info("Train: %d, Dev: %d, Test: %d",
                     len(data_dict["train"]), len(data_dict["dev"]), len(data_dict["test"]))
        return data_dict

    def run(self):
        score, test_preds_df = self.train_and_test_on_splits(
            train=self.data_dict["train"],
            dev=self.data_dict["dev"],
            test=self.data_dict["test"],
        )
        score = pd.DataFrame(score)
        return score, test_preds_df

    def train_and_test_on_splits(self, train, dev, test):
        logger.info("Train shape: %s, Dev shape: %s, Test shape: %s",
                     train.shape, dev.shape, test.shape)
        scores = []
        train, dev, test = self.encode_values(train.copy(), dev.copy(), test.copy())

        logger.info("Language model: %s", self.params.language_model_name)
        model = self._new_model(train_df=train)
        train_dataset = self.get_batches(train)
        dev_dataset = self.get_batches(dev)
        logger.info("Params: %s", self.get_param_combinations(sep1=": "))
        self.print_embs_info(model)

        # Robust step calculation: prevent zero steps on small datasets
        epoch_steps = max(1, math.ceil(len(train) / self.params.batch_size))
        save_eval_steps = max(1, epoch_steps // 2)
        logger.info("Epoch steps: %d, save/eval steps: %d", epoch_steps, save_eval_steps)

        param_combinations = self.get_param_combinations(
            exclude_list=[
                "balance_annotator_weights",
                "data_name",
                "approach",
                "embedding_colnames",
                "max_len",
            ]
        )
        saving_models_dir = str(
            REPO_ROOT / "saved_models" / self.params.approach
            / f"{self.params.data_name}_{self.params.embedding_colnames}"
            / param_combinations
        )
        training_args = self.get_trainingargs(
            num_save_eval_log_steps=save_eval_steps,
            saving_models_dir=saving_models_dir,
        )

        trainer = self.get_trainer(
            model=model,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            training_args=training_args,
        )
        logger.info("Trainable parameters: %d", trainer.get_num_trainable_parameters())
        trainer.train()

        self.print_embs_info(model)

        logger.info("Dev predictions (individually):")
        dev_preds = trainer.predict(dev_dataset)
        scores_dict = {k[5:]: v for k, v in dev_preds.metrics.items()}

        scores_dict["type"] = "dev"
        dev = self.add_predictions(df=dev, preds=dev_preds)
        (
            dev,
            scores_dict["corr_disagreement"],
            scores_dict["avg_disagreement_labels"],
            scores_dict["avg_disagreement_preds"],
        ) = self.calculate_disagreement(df=dev)

        scores_dict["rand_seed"] = self.params.random_state
        logger.info("Dev scores: %s", scores_dict)
        scores.append(scores_dict)

        if not self.params.skip_test:
            scores_test, test_df = self.run_tests(
                trainer,
                train,
                test,
            )
            test_df["rand_seed"] = self.params.random_state
            scores = scores + scores_test
        else:
            test_df = None
            logger.info("Removing saved models at: %s", saving_models_dir)
            import shutil
            shutil.rmtree(saving_models_dir, ignore_errors=True)

        return scores, test_df

    def get_trainer(self, model, train_dataset, dev_dataset, training_args):
        from transformers import Trainer

        training_args.label_names = ["labels"]
        return Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=self.tokenizer,
            args=training_args,
            compute_metrics=self.compute_metrics_function,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.params.early_stopping_patience,
                    early_stopping_threshold=0.01,
                )
            ],
        )

    def get_trainingargs(self, num_save_eval_log_steps, saving_models_dir):
        metric_for_best_model = "eval_macro_f1"
        training_args = {
            "output_dir": saving_models_dir,
            "evaluation_strategy": "steps",
            "eval_steps": num_save_eval_log_steps,
            "logging_strategy": "steps",
            "logging_steps": num_save_eval_log_steps,
            "load_best_model_at_end": True,
            "metric_for_best_model": metric_for_best_model,
            "greater_is_better": True,
            "save_strategy": "steps",
            "save_steps": num_save_eval_log_steps,
            "save_total_limit": 2,
            "num_train_epochs": self.params.num_epochs,
            "per_device_train_batch_size": self.params.batch_size,
            "per_device_eval_batch_size": self.params.batch_size,
            "learning_rate": self.params.learning_rate,
            "weight_decay": 0.01,
            "warmup_steps": 2 * num_save_eval_log_steps,
            "remove_unused_columns": False,
            "seed": self.params.random_state,
            "label_names": self.task_labels,
        }

        training_args = MyTrainingArguments(**training_args)
        return training_args

    @abc.abstractmethod
    def get_annotators(self, df):
        """Return a list of unique annotator identifiers from the dataframe."""

    @abc.abstractmethod
    def get_batches(self, df):
        """Convert a dataframe into a tokenized HuggingFace Dataset."""

    @abc.abstractmethod
    def _new_model(self, train_df):
        """Create and return a new model instance."""

    @abc.abstractmethod
    def _create_loss_label_weights(self, data):
        """Compute per-label loss weights."""

    def calculate_disagreement(self, df):
        df["continuous_disagreement_labels"] = self.calculate_continous_disagreements(
            df=df, label_or_pred_col="label"
        )
        df["continuous_disagreement_preds"] = self.calculate_continous_disagreements(
            df=df, label_or_pred_col="pred"
        )
        temp = df.drop_duplicates(self.instance_id_col)
        corr_disagreement = temp["continuous_disagreement_preds"].corr(
            temp["continuous_disagreement_labels"], method="pearson"
        )
        return (
            df,
            corr_disagreement.round(3),
            temp["continuous_disagreement_labels"].mean().round(3),
            temp["continuous_disagreement_preds"].mean().round(3),
        )

    def run_tests(self, trainer, train, test):
        """
        Gets the trained model, reports performance based on:
            1) majority label vs majority pred
            2) annotators macro or micro performance
        """
        preds = trainer.predict(self.get_batches(test))
        test = self.add_predictions(df=test, preds=preds)
        scores = self._calculate_annotator_performance(train=train, test=test)
        (
            test,
            scores[-1]["corr_disagreement"],
            scores[-1]["avg_disagreement_labels"],
            scores[-1]["avg_disagreement_preds"],
        ) = self.calculate_disagreement(test)
        logger.info("Test scores (individually): %s", scores[-1])

        if self.params.majority_inference:
            maj_scores_dict, test_expanded_results = (
                self._calculate_majority_performance(
                    trainer=trainer, test=test, train=train
                )
            )
            logger.info("Test majority scores: %s", maj_scores_dict)
            scores.append(maj_scores_dict)
            assert not test_expanded_results.empty
            return scores, test_expanded_results

        return scores, test

    def tokenize_batch(self, batch: dict) -> dict:
        """Batched tokenization function for use with Dataset.map(batched=True)."""
        if self.instance_id_col == "pair_id":
            return self.tokenizer(
                text=batch["parent_text"],
                text_pair=batch["text"],
                padding="max_length",
                truncation=True,
                max_length=self.params.max_len,
            )
        return self.tokenizer(
            text=batch["text"],
            padding="max_length",
            truncation=True,
            max_length=self.params.max_len,
        )

    def print_embs_info(self, model):
        pass
