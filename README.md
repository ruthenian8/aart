### This repository contains the code for the paper: [Hypernetworks for Perspectivist Adaptation](#).

The code in this repository extends the abstractions from AART: (https://github.com/negar-mokhberian/aart).

If you copy or use the code, consider citing both our paper and AART.

---

### Installation

Ensure you have Python installed. Then, clone this repository and install the required dependencies:

```bash
git clone https://github.com/ruthenian8/Hypernets.git
cd aart
pip install -r requirements.txt
```

---

## Running

You can explore available command-line arguments using:

```bash
python main.py --help
```

**Example Run Command:**
```bash
python main.py --data_name my_dataset --approach NHW
```

- `--data_name`: A custom name for your dataset.
- `--approach`: `"HPM"`.

---

## Dataset Format

The dataset should be stored under:

```
./data/APPROACH/DATA_NAME/all_data.csv
```

where:
`APPROACH` and `DATA_NAME` are both provided as input arguments. 
- `APPROACH` corresponds to the selected method (`single`, `multi_task`, or `aart` or `hpm`).
- `DATA_NAME` is a user-defined dataset name.

### Expected Columns

| Column Name   | Description |
|--------------|------------|
| **prep_text**  | Preprocessed texts. Apply your preferred preprocessing method before storing |
| **text_id**    | A unique numerical ID for each text instance |
| **annotator**  | A unique identifier for each annotator (e.g., `annotator_0`, `annotator_1`, ...) |
| **label**      | The annotation provided by the respective annotator for the given text |

---
