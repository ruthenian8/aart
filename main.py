import os
import re
import logging
from datetime import datetime
from pathlib import Path

import pytz

from params import Params
from pipelines.hpm_pipeline import HPMPipeline

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent


def slugify_experiment_name(name: str, max_length: int = 200) -> str:
    """Sanitize a string for safe use in filesystem paths."""
    name = re.sub(r"[/:,\s]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name[:max_length]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hypernetworks for Perspectivist Adaptation (HPM)"
    )

    parser.add_argument(
        "--data_name", default=None, type=str, required=True, help="The corpus name"
    )

    parser.add_argument(
        "--language_model_name",
        default=None,
        type=str,
        required=True,
        help="The transformer model name, e.g. roberta-base",
    )

    parser.add_argument(
        "--approach",
        default=None,
        type=str,
        required=True,
        choices=["hpm"],
        help="The modeling approach. Currently only 'hpm' is supported.",
    )

    parser.add_argument(
        "--max_len", type=int, help="Maximum sentence length after tokenizing"
    )

    parser.add_argument("--batch_size", type=int, help="the batch size for training")

    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate for training the model."
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="maximum number of epochs during the early stopping",
    )

    parser.add_argument("--random_state", default=2023, type=int)

    parser.add_argument("--skip_test", action="store_true")

    parser.add_argument(
        "--majority_inference",
        action="store_true",
        help="whether or not to infer the majority vote from the trained model",
    )

    parser.add_argument(
        "--lambda2",
        default=0.0,
        type=float,
        help="coefficient for l2 regularizing the annotator embeddings",
    )

    parser.add_argument(
        "--contrastive_alpha",
        default=0,
        type=float,
        help="coefficient for contrastive loss for annotator embeddings",
    )

    parser.add_argument(
        "--embedding_colnames",
        default="",
        type=str,
        help="comma separated string of columns to make embeddings for, e.g. annotator,race,age...",
    )

    parser.add_argument(
        "--sort_instances_by",
        default="",
        type=str,
        help="If provided then the instances will be sorted by this basis and there won't be shuffling of training instances after each epoch",
    )

    parser.add_argument(
        "--num_fake_annotators",
        type=int,
        default=0,
        help="number of fake annotators to add.",
    )

    args = parser.parse_args()
    # Normalize approach to lowercase
    args.approach = args.approach.lower()
    return args


def get_pipeline(params):
    if params.approach == "hpm":
        return HPMPipeline(params)
    raise ValueError(
        f"Unsupported approach '{params.approach}'. Only 'hpm' is currently supported."
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    main_args = parse_args()
    params = Params.from_namespace(main_args)
    pipeline = get_pipeline(params)
    logger.info("Pipeline: %s", pipeline)

    #############################################################################
    ############################## Experiments ##################################
    #############################################################################

    score, test_preds_df = pipeline.run()

    #############################################################################
    ################################ Results ####################################
    #############################################################################
    # saving the results along with the hyper parameters of the model
    score["params"] = pipeline.get_param_combinations(
        ": ", exclude_list=["random_state", "balance_annotator_weights", "lambda1"]
    )
    score["rand_seed"] = params.random_state
    score["approach"] = params.approach

    pacific = pytz.timezone("US/Pacific")
    sa_time = datetime.now(pacific)
    name_time = sa_time.strftime("%m%d%y-%H%M")
    score["time"] = name_time

    results_dir = REPO_ROOT / "results" / params.approach / params.data_name
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving results to %s", results_dir)

    if params.skip_test:
        scores_file = results_dir / f"dev_scores_classification_{params.approach}.csv"
    else:
        pred_name = slugify_experiment_name(
            f"predictions_{name_time}_{params.approach}_{params.random_state}_{params.embedding_colnames}"
        )
        test_predictions_path = results_dir / f"{pred_name}.csv"
        logger.info("predictions path: %s", test_predictions_path)
        test_preds_df.to_csv(test_predictions_path, index=False)
        scores_name = slugify_experiment_name(
            f"scores_classification_{params.approach}_{params.embedding_colnames}"
        )
        scores_file = results_dir / f"{scores_name}.csv"

    logger.info("Scores path: %s", scores_file)

    # the scores of each test is appended as a row to the scores file
    if scores_file.exists():
        score.to_csv(scores_file, header=False, index=False, mode="a")
    else:
        score.to_csv(scores_file, index=False)


if __name__ == "__main__":
    main()
