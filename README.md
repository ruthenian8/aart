# Hypernetworks for Perspectivist Adaptation (HPM)

This repository contains the code for the paper: [Hypernetworks for Perspectivist Adaptation](#).

The code extends abstractions from AART: (https://github.com/negar-mokhberian/aart).

If you copy or use the code, consider citing both our paper and AART.

---

## Installation

Ensure you have Python 3.9+ installed. Then, clone this repository and install the required dependencies:

```bash
git clone https://github.com/ruthenian8/Hypernets.git
cd Hypernets
pip install -r requirements.txt
```

---

## Supported Models

**Current backbone support:** RoBERTa-style models with `query`/`value` attention modules (e.g. `roberta-base`, `cardiffnlp/twitter-roberta-base-offensive`).

Other transformer families may work if they expose compatible attention projection modules, but have not been tested.

---

## Running

You can explore available command-line arguments using:

```bash
python main.py --help
```

**Example Run Command:**
```bash
python main.py \
    --data_name my_dataset \
    --approach hpm \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 20 \
    --embedding_colnames annotator \
    --max_len 128 \
    --language_model_name roberta-base
```

- `--data_name`: A custom name for your dataset.
- `--approach`: The modeling approach. Currently only `hpm` is supported.
- `--majority_inference`: Pass this flag to infer majority vote from the trained model at test time.

---

## Dataset Format

The dataset should be stored under:

```
./data/hpm/DATA_NAME/all_data.csv
```

where `DATA_NAME` is provided as an input argument.

Train/dev/test splits should be stored under:

```
./splits/DATA_NAME/train_{random_state}.txt
./splits/DATA_NAME/dev_{random_state}.txt
./splits/DATA_NAME/test_{random_state}.txt
```

### Expected Columns

| Column Name   | Description |
|--------------|------------|
| **prep_text**  | Preprocessed texts. Apply your preferred preprocessing method before storing |
| **text_id**    | A unique numerical ID for each text instance |
| **annotator**  | A unique identifier for each annotator (e.g., `annotator_0`, `annotator_1`, ...) |
| **label**      | The annotation provided by the respective annotator for the given text |

For pair datasets, also include:
- **pair_id**: A unique numerical ID for each text pair
- **prep_parent_text**: The preprocessed parent text

---

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

---
