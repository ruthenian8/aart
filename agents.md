# agents.md

## Objective

Refactor the repository to preserve the current **hypernetwork + LoRA + annotator-conditioning pipeline** while improving:

1. correctness and runtime stability,
2. training/inference efficiency,
3. portability across environments,
4. maintainability and implementation clarity.

The intent is **not** to change the research idea or the top-level behavior of the HPM pipeline. The intent is to remove accidental complexity, fix real bugs, and make future work safer.

---

## Ground rules for the refactor

1. **Preserve the current modeling idea**:
   - pretrained sequence classifier backbone,
   - LoRA adapters on attention projections,
   - hypernetwork-generated annotator-specific adapter weights,
   - per-annotator and majority-vote evaluation.

2. **Do not silently change the experimental protocol**:
   - keep the current split semantics,
   - keep the current label semantics,
   - keep the current majority-vote tie-break behavior unless it is explicitly documented and tested,
   - keep default LoRA rank/alpha/dropout unless they are only being surfaced as config.

3. **Prefer low-risk refactors first**. Anything that could affect model outputs should be implemented behind a test that demonstrates parity on a fixed seed and a synthetic mini-batch.

4. **Treat HPM as the only supported path unless additional approaches are actually implemented**. The repo currently exposes other approaches in the CLI, but the code path is only wired for HPM.

---

## Executive summary of required changes

### Must-fix correctness issues

1. **Fix broken device initialization and hard-coded CUDA assumptions.**
2. **Fix broken majority-inference path** (currently fails when labels are absent and also handles predictions inconsistently).
3. **Fix brittle path handling** for data/splits/results so execution does not depend on the current working directory.
4. **Fix CLI/documentation mismatch** (`hpm` vs `HPM`, invalid examples, unsupported approaches exposed in parser).
5. **Fix training-step calculations** so small datasets do not produce zero eval/save/logging steps.
6. **Fix or remove dead code paths that are currently invalid** (`AdaptedLinear`, unused loss-weight plumbing, unused imports, placeholder methods without abstract enforcement).

### High-value efficiency improvements

1. **Replace per-example tokenization with batched tokenization.**
2. **Stop running the base model once per sample when annotator IDs repeat inside a batch.** Group by annotator ID at minimum.
3. **Remove repeated device transfers and repeated recomputation where possible.**

### Maintainability improvements that should accompany the above

1. Introduce a proper config object with validation.
2. Replace print-based tracing with structured logging.
3. Sanitize experiment/run names used in filesystem paths.
4. Add smoke tests and regression tests for the HPM path.

---

## Detailed implementation plan by file

---

## 1) `main.py`

### Current issues

- `parse_args()` exposes `single`, `multi_task`, `aart`, and `hpm`, but `get_pipeline()` only returns a pipeline for `hpm`.
- Example command lines in the repo do not match the parser behavior.
- Results paths are relative to the current working directory.
- The code imports `numpy` but does not need it except for `np.nan` defaults in argparse.

### Required changes

#### 1.1 Restrict or validate the supported approach

Current behavior is misleading. There are two defensible options:

**Preferred option:**
- Change the parser to accept only `hpm` for now.
- Update help text accordingly.

**Alternative option:**
- Keep the other choices, but make `get_pipeline()` raise a clear `NotImplementedError` with the exact unsupported approach name.

Do **not** leave the current silent `None` return.

#### 1.2 Normalize the approach casing

- Accept `hpm` only, or normalize user input with `.lower()` before validation.
- Update all example scripts accordingly.

#### 1.3 Make all repo paths root-relative

Introduce a repo-root helper using `pathlib.Path`:

- `REPO_ROOT = Path(__file__).resolve().parent`
- Use `REPO_ROOT / "results" / ...`

This prevents failures when the script is launched from a different directory.

#### 1.4 Clean up result naming

The current filename construction can include commas, spaces, and other path-hostile characters from parameter strings.

Implement a helper such as:

- `slugify_experiment_name(str) -> str`

Requirements:
- replace `/`, `:`, spaces, commas, and repeated separators,
- cap path segment length,
- preserve enough information for reproducibility.

### Acceptance criteria

- Running `python main.py --approach hpm ...` resolves the correct pipeline.
- Running with an unsupported approach produces a clear error.
- Output files are written correctly regardless of the current working directory.

---

## 2) `README.md` and `test_run.sh`

### Current issues

- README says `cd aart`, which does not match this repo.
- README example uses `NHW`, while parser expects `hpm`.
- `test_run.sh` uses `--approach "HPM"`, which is invalid against the current parser.
- README references `requirements.txt`, but the repo currently does not include one.
- README suggests model-name flexibility, but the implementation is RoBERTa-specific.

### Required changes

#### 2.1 Fix installation and run instructions

- Replace `cd aart` with the actual repo directory.
- Replace all examples with `--approach hpm`.
- Remove references to nonexistent approaches unless they are actually implemented.

#### 2.2 Add an explicit support matrix

Document what is truly supported now:

- **Backbone support (current reality):** RoBERTa-style models with `query`/`value` attention modules.
- If non-RoBERTa models are not yet generalized, say so explicitly.

#### 2.3 Add `requirements.txt`

At minimum include the direct runtime dependencies actually used by the HPM path:

- torch
- transformers
- peft
- datasets
- pandas
- numpy
- scikit-learn
- pytz
- optionally umap-learn, seaborn, matplotlib if plotting is kept

Pin reasonably, but avoid over-constraining unless tested.

### Acceptance criteria

- README commands run without obvious argument mismatches.
- `test_run.sh` uses a valid approach value.
- A new user can install dependencies from the repo itself.

---

## 3) `pipelines/generic_pipeline.py`

This file has the highest concentration of stability issues.

### Current issues

1. **Broken CUDA diagnostics** in `__init__`:
   - calls `torch.cuda.current_device()` unconditionally,
   - calls `torch.cuda.get_device_name(id)` where `id` is the Python builtin, not a GPU index,
   - assumes CUDA is always present.

2. **Determinism settings are contradictory**:
   - `torch.backends.cudnn.benchmark = True`
   - `torch.backends.cudnn.deterministic = True`

3. **Path handling is brittle**:
   - split files are read from `../splits/...` relative to the working directory.

4. **Placeholder methods use `pass` instead of abstract enforcement**.

5. **Tokenization is inefficient**:
   - per-example map with `num_proc=16`,
   - lambda-based mapping,
   - in-memory token cache that is not effective when multiprocessing is used.

6. **Step calculation can become zero**:
   - `epoch_steps = int(train.shape[0] / batch_size)`
   - `eval_steps = int(epoch_steps / 2)`

7. **Logging is noisy and not structured**.

### Required changes

#### 3.1 Introduce a single device resolver

Create a helper on pipeline init:

- `self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

Then:
- only print CUDA properties if CUDA is available,
- use `torch.cuda.current_device()` only inside the CUDA branch,
- fix the invalid `get_device_name(id)` call.

If CPU execution is not intended to be fast, that is fine. It still must fail gracefully or run correctly for smoke tests.

#### 3.2 Fix seeding behavior

For deterministic mode:
- set `benchmark = False` when `deterministic = True`.

Recommended implementation:
- add a `deterministic` flag to config,
- default to deterministic for reproducibility.

#### 3.3 Use `pathlib.Path` everywhere

Replace string path building in:
- `read_data()`
- model save paths
- embedding output paths
- results paths

with `Path` joins rooted at the repo root.

#### 3.4 Make `GenericPipeline` an abstract base class

Use `abc.ABC` and `@abstractmethod` for:
- `get_annotators`
- `get_batches`
- `_new_model`
- `_create_loss_label_weights`

This prevents partially implemented pipelines from failing later and more opaquely.

#### 3.5 Replace per-example tokenization with batched tokenization

Current code:
- uses `Dataset.map(lambda x: self.tokenize_function(x), num_proc=16)`
- caches tokenizations in Python dictionaries keyed by raw text

Problems:
- slow,
- cache is ineffective with multiprocessing,
- memory-heavy,
- harder to reason about.

Refactor as follows:

- make tokenization batched: `map(self.tokenize_batch, batched=True, batch_size=..., num_proc=...)`
- remove `self.tokenizations` entirely unless benchmarking proves it is useful,
- in the pair case, tokenize `text` and `text_pair` in batch form.

Suggested API:

```python

def tokenize_batch(self, batch: dict) -> dict:
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
```

This is the cleanest efficiency improvement that does not change model behavior.

#### 3.6 Make save/eval/log step counts robust

Replace:
- `epoch_steps = int(train.shape[0] / self.params.batch_size)`
- `int(epoch_steps / 2)`

with something like:
- `epoch_steps = max(1, math.ceil(len(train_dataset) / batch_size))`
- `save_eval_steps = max(1, epoch_steps // 2)`

This prevents invalid Hugging Face `TrainingArguments` on small datasets.

#### 3.7 Use logging instead of print spam

Implement a module-level logger and reserve info/debug logs for:
- dataset sizes,
- resolved device,
- experiment path,
- metric summaries.

Do not log giant argument dumps every time unless debug mode is enabled.

### Acceptance criteria

- Pipeline initialization works on CPU-only machines without crashing in diagnostics.
- Tokenization throughput improves materially on the same dataset.
- Small datasets no longer produce zero `eval_steps` or `save_steps`.
- The abstract base class catches incomplete pipeline implementations early.

---

## 4) `pipelines/hpm_pipeline.py`

This file contains both important logic and several correctness bugs.

### Current issues

1. **Majority-inference path is broken** for multiple reasons:
   - `test_dataset_expanded.remove_columns("labels")` removes labels,
   - `HyperLoRAModel.forward()` assumes labels are always present,
   - `preds_expanded.predictions[0][:, 1:]` is inconsistent with the non-majority path and is very likely wrong.

2. **`expand_test()` is incomplete for pair datasets**:
   - it only preserves `prep_text`, not `prep_parent_text`.

3. **`get_batches()` uses fixed `num_proc=16`**, which is wasteful and brittle.

4. **Encoding logic is fragile**:
   - special-case filtering for `emfd` is hard-coded into the main path,
   - unknown categories in dev/test are handled inconsistently.

5. **Loss-weight computation is currently unused**:
   - `_create_loss_label_weights()` computes per-annotator weights,
   - `HyperLoRAModel.forward()` does not use them.

6. **Hard-coded CUDA in tensor creation**.

### Required changes

#### 4.1 Fix majority inference end-to-end

Implement three coordinated changes:

**A. Make model forward support inference without labels**
- In `HyperLoRAModel.forward()`, treat `labels` as optional.
- If labels are absent, return logits only and set `loss=None`.

**B. Normalize prediction extraction**
- Create one helper that always extracts logits from `Trainer.predict()` output.
- Use it in both `add_predictions()` and `_calculate_majority_performance()`.

Example helper contract:

```python

def extract_model_logits(predictions) -> np.ndarray:
    raw = predictions.predictions
    if isinstance(raw, tuple):
        raw = raw[0]
    return raw
```

Then always strip the prepended annotator ID in exactly one place:

```python
logits = extract_model_logits(preds)
class_logits = logits[:, 1:]
preds = class_logits.argmax(axis=1)
```

**C. Ensure expanded test batches contain all required text fields**
- If `self.instance_id_col == "pair_id"`, carry both `prep_parent_text` and `prep_text` into the expanded dataframe.

This entire path must be covered by a regression test.

#### 4.2 Make device usage pipeline-owned

Replace all `device="cuda"` and `torch.device("cuda")` literals with `self.device` or an injected model device.

Specifically update:
- `_create_loss_annotator_weights()`
- `_create_loss_label_weights()`
- `_new_model()`

#### 4.3 Remove hard-coded EMFD logic from the main encoding path

Current code silently filters dev/test rows for a specific dataset name substring.
That is too implicit.

Replace with one of these explicit policies:

- **Preferred:** introduce `unknown_annotator_policy` with allowed values such as `error`, `drop`, `map_to_unknown`.
- Default to `error` or `drop` depending on the research intent, but make the behavior explicit and logged.

This is important because hidden data dropping can change evaluation sets without obvious visibility.

#### 4.4 Make `get_batches()` use configurable multiprocessing

Expose tokenization workers as config, e.g. `num_proc`, with a safe default:
- `None`, `0`, or `min(os.cpu_count(), 4)` depending on tested behavior.

Do not hard-code `16`.

#### 4.5 Resolve unused loss-weight plumbing

There are two defensible options. Pick one and document it:

**Preferred option for parity:**
- remove the unused per-annotator loss-weight computation from the HPM path,
- remove `loss_weights` from the model constructor,
- remove the dead code comments around weighted CE.

**Alternative option:**
- implement class-weighted CE correctly with full `num_labels`-length tensors,
- ensure it is optional and defaults to current behavior.

Do **not** keep the current half-implemented state.

### Acceptance criteria

- `majority_inference=True` runs end-to-end.
- Pair datasets work in both standard and majority-inference paths.
- Unknown annotator/category handling is explicit and documented.
- HPM no longer hard-codes CUDA.

---

## 5) `model_architectures.py`

This file contains the core performance bottleneck.

### Current issues

1. **Forward is serial over batch items**:
   - hypernetwork outputs are generated for the whole batch,
   - but the base model is still run once per sample inside a Python loop,
   - hooks are registered and removed per sample.

2. **The implementation is RoBERTa-specific**:
   - assumes `self.model.base_model.roberta.encoder.layer` exists.

3. **Repeated device transfers inside the hot path**:
   - `wA = A[j].to(self.device)` and `wB = B[j].to(self.device)` inside the context-manager loop.

4. **Several imports are unused**:
   - `re`, `PreTrainedModel`, `Any`, `List`, `Tuple`, `AdaptedLinear`, `LabelEncoder`.

5. **`labels` are mandatory even for pure inference**, which breaks majority inference.

### Required changes

#### 5.1 Immediate performance refactor: group batch items by annotator ID

This is the safest high-impact refactor.

Observation:
- hypernetwork conditioning depends on `annotator_ids`,
- if multiple examples in a batch share an annotator, they can share the same generated LoRA weights.

Refactor `forward()` to:

1. compute unique annotator IDs in the batch,
2. for each unique annotator ID:
   - get all batch indices for that annotator,
   - generate one set of LoRA weights,
   - run the base model once on the whole sub-batch,
3. scatter the resulting logits back into the original batch order.

This preserves the current mathematics while removing a large amount of repeated work.

This should be implemented before any more invasive vectorization.

#### 5.2 Optional phase-2 optimization: replace hook-based injection with explicit dynamic LoRA modules

The current hook mechanism is clever but expensive and fragile.
A more robust design is:

- subclass or wrap the PEFT-inserted LoRA linear layers,
- pass annotator-conditioned A/B tensors to those layers directly,
- apply the base linear projection plus low-rank delta in one forward pass.

Only do this after the grouped-by-annotator refactor has tests proving parity.
It is a bigger change and should not be the first step.

#### 5.3 Generalize LoRA module discovery

Replace hard-coded RoBERTa traversal with module discovery that finds PEFT-injected LoRA modules generically.

Acceptable approaches:
- iterate through `self.base_model.named_modules()` and collect modules that expose `lora_A` and `lora_B`,
- optionally filter by configured target module names.

This lets the code honestly support more than one transformer family, provided the target module names are valid.

If this is not implemented now, then **narrow the README claims** and enforce RoBERTa-only backbones in validation.

#### 5.4 Make `labels` optional in forward

Required for inference-only prediction paths.

Implementation contract:
- if labels are present: return `{"loss": loss, "logits": catted_logits}`
- if labels are absent: return `{"logits": catted_logits}`

#### 5.5 Remove unused imports and dead comments

This file should be reduced to the imports it actually uses.

### Acceptance criteria

- Batch forward no longer runs the backbone once per individual example when annotator IDs repeat.
- Majority inference works without labels.
- Module discovery is either generalized or the code explicitly validates that the selected backbone is RoBERTa-compatible.

---

## 6) `model_hypernetwork.py`

### Current issues

- The file is comparatively clean, but there are still small improvements:
  - `get_context_embeddings()` recreates the index tensor every call,
  - context embeddings are effectively static identifiers and can be handled more efficiently.

### Required changes

#### 6.1 Avoid repeated context-index construction

Register a buffer for module indices if the number of modules is fixed:

- `self.register_buffer("module_indices", torch.arange(num_modules), persistent=False)`

Then use that buffer in `get_context_embeddings()`.

This is small, but it removes needless tiny allocations from a hot path.

#### 6.2 Add shape assertions in debug mode

Add optional assertions after reshaping A/B outputs:
- `(batch, num_mod, r, in_dim)`
- `(batch, num_mod, out_dim, r)`

This is helpful when changing target modules or LoRA rank.

### Acceptance criteria

- No behavioral change.
- Slightly cleaner and safer hypernetwork forward path.

---

## 7) `model_adapted_linear.py`

### Current issues

This file appears to be dead code in the current HPM path, and its forward logic is not valid in its current form.

Concrete problems:
- uses a weak reference to the hypernetwork, which can become `None` if no strong reference exists,
- reshape logic for `x.view(batch_size, self.in_features, x.size(1))` is invalid for a standard 2D input,
- `out.squeeze(2).permute(0, 2, 1)` is incompatible with the preceding tensor shape.

### Required changes

Choose one option:

**Preferred option:**
- remove this file from the active code path entirely,
- add a short comment in the repo history or docs that it is an experimental stub not used by HPM.

**Alternative option if it must stay:**
- move it into an `experimental/` namespace,
- add a unit test proving the shape math for a realistic input tensor,
- replace weakref usage with explicit ownership or dependency injection.

Do **not** leave it in the main module namespace looking production-ready when it is not.

### Acceptance criteria

- No dead, broken implementation remains in the main HPM path.

---

## 8) `params.py`

### Current issues

- class name is lowercase and non-idiomatic,
- `update()` mutates and deletes attributes dynamically,
- broad `except:` hides real errors,
- config validation is weak.

### Required changes

#### 8.1 Replace with a dataclass config object

Create a proper `Params` dataclass with explicit types and defaults.

Suggested additions:
- `device: Optional[str] = None`
- `num_tokenization_workers: Optional[int] = None`
- `deterministic: bool = True`
- `unknown_annotator_policy: str = "error"`
- `backbone_family: Optional[str] = None` if needed for validation

#### 8.2 Add a constructor from argparse namespace

Implement:
- `Params.from_namespace(args)`

and validate:
- supported approach,
- positive batch size,
- positive max length,
- nonnegative epochs,
- sane tokenization worker count.

#### 8.3 Remove dynamic attribute deletion

The current pattern of deleting attributes depending on approach is brittle.
Instead:
- keep optional fields on the config object,
- validate which ones are used for which approach,
- ignore unused ones explicitly.

### Acceptance criteria

- Config is explicit, typed, and validated.
- No broad `except:` is used for routine config handling.

---

## 9) `utils.py`

### Current issues

- metric function selection is based on a string match over `type(pipeline_obj)`,
- macro F1 can become `NaN` if no annotator has more than 5 examples.

### Required changes

#### 9.1 Make metric dispatch explicit

Replace:
- `if "HPM" in str(type(pipeline_obj)):`

with one of:
- `if pipeline_obj.params.approach == "hpm":`
- or a direct callable passed from the pipeline.

#### 9.2 Make macro-F1 robust

When no annotator satisfies the threshold, return a defined fallback:
- either `0.0`,
- or the micro F1,
- or raise a clear exception during evaluation.

The fallback must be documented because `eval_macro_f1` is used for early stopping.

### Acceptance criteria

- Metric selection is explicit.
- `macro_f1` cannot silently become `NaN` and destabilize model selection.

---

## 10) Test coverage to add before and during refactor

Add a minimal test suite. This is necessary because several changes above touch control flow in training/inference.

### 10.1 Unit tests

#### A. CLI/config tests
- parser accepts `hpm` and rejects invalid casing if case sensitivity is intended,
- unsupported approaches fail clearly.

#### B. Device/init tests
- pipeline init does not crash on CPU-only environments,
- path resolution is independent of current working directory.

#### C. Tokenization tests
- single-text datasets tokenize correctly,
- pair datasets tokenize correctly,
- batched tokenization output keys match the previous implementation.

#### D. Model forward tests
- `HyperLoRAModel.forward()` works with labels,
- `HyperLoRAModel.forward()` works without labels,
- output shape is `(batch, 1 + num_labels)` because annotator ID is prepended.

#### E. Majority inference tests
- `_calculate_majority_performance()` runs on a tiny synthetic dataset,
- pair-dataset version also runs.

### 10.2 Regression/parity tests

These are important for defensibility.

#### A. Prediction parity for low-risk refactors
For the following changes, prove parity on a fixed seed and small synthetic batch:
- batched tokenization refactor,
- grouped-by-annotator forward refactor,
- path/config/logging cleanups.

#### B. Metric parity
On a frozen synthetic prediction tensor, confirm that:
- `add_predictions()` and majority-inference prediction extraction use the same logic,
- micro/macro metrics match expected values.

### 10.3 Smoke test

Add one end-to-end smoke test with a very small synthetic dataset that exercises:
- train/dev/test split loading,
- HPM training for 1 epoch,
- standard prediction,
- majority inference.

---

## 11) Recommended implementation order

Follow this order to reduce risk.

### Phase 1: correctness and portability
1. Fix CLI/docs/test script mismatch.
2. Add `requirements.txt`.
3. Introduce repo-root path handling.
4. Fix device initialization and remove hard-coded CUDA assumptions.
5. Fix step-count calculation.
6. Make `HyperLoRAModel.forward()` support no-label inference.
7. Fix majority-inference prediction extraction.
8. Fix pair-dataset handling in `expand_test()`.

### Phase 2: low-risk efficiency gains
9. Replace tokenization with batched tokenization.
10. Remove ineffective token cache.
11. Group forward passes by unique annotator ID inside each batch.
12. Remove repeated `.to(self.device)` calls in the hot path when tensors are already on-device.

### Phase 3: maintainability cleanup
13. Replace `params` with validated `Params` dataclass.
14. Make `GenericPipeline` abstract.
15. Remove dead imports and dead code.
16. Resolve the unused `loss_weights` plumbing.
17. Move or delete `AdaptedLinear`.
18. Introduce structured logging.

### Phase 4: optional generalization
19. Generalize LoRA module discovery across backbone families.
20. Only then broaden README model-support claims.

---

## 12) What should not be changed unless explicitly re-baselined

To preserve the current HPM behavior, do **not** change these during the refactor unless there is a deliberate experiment re-baseline:

- LoRA rank default,
- LoRA target modules (`query`, `value`) unless support is being generalized,
- majority-vote aggregation rule,
- label encoding semantics,
- default tokenizer max length unless only surfaced as config,
- train/dev/test split definitions.

If any of these are changed, the agent must document the reason and rerun the parity/smoke tests.

---

## 13) Final deliverable expectations

The refactored repo should satisfy all of the following:

1. `python main.py --approach hpm ...` is the only documented and working mainline path.
2. The code runs from any working directory inside or outside the repo.
3. CPU-only environments can at least execute smoke tests without crashing during init.
4. Majority inference works.
5. Pair datasets work in both standard and majority paths.
6. Tokenization is batched and measurably faster.
7. The model forward path no longer performs one full backbone call per individual example when repeated annotator IDs exist in a batch.
8. README instructions are accurate.
9. The repo contains a dependency file and basic automated tests.
10. All removed or altered behavior is justified in code comments and/or PR notes.

---

## Short rationale for why these changes are defensible

These changes are defensible because they do **not** alter the core HPM research idea. They address:

- clearly broken code paths,
- misleading documentation,
- environment-specific crashes,
- wasted computation,
- hidden assumptions that currently make the repo fragile.

The refactor should make the repository easier to trust, easier to reproduce, and cheaper to run, while keeping the same conceptual hypernetwork pipeline.
