# Local VLM – pipeline runbook

From a taxonomy change down to training, evaluation and release.
Paths are relative to `~/projects/local-vlm-project` (WSL) unless the step says Windows.
Last updated 2026-11-24 (state: shoes+bags frozen as `shoes-bags-v1`; women's taxonomy clean locally and on Heroku; RTW next). Sections: 0 conventions · 1 taxonomy (1b removed subcategories) · 2 WSL alignment · 3 guidelines · 4 new batch · 5 label fixes · 6 dataset · 7 training · 8 eval · 9 fast experiment · 10 release · 11 app export · 12 housekeeping · 13 diagnostic tools.

---

## 0. Conventions

| Thing | Rule |
|---|---|
| Ground truth | Our labels in `data/raw/labels/*.json`. Filenames are never a label hint. |
| One adapter = its own files | `configs/dataset_manifest_<adapter>.yaml`, `configs/model_<adapter>.yaml`. Always pass them explicitly. Experiments live in `configs/experiments/`. |
| Adapter registry | `configs/adapters.yaml`: path, version, report, git tag, manifest, model config, taxonomy fingerprint, scores (a record; nothing reads it yet). |
| Frozen adapters | Never edit their guidelines without retraining. For a shared file (`attribute_values.yaml`, `_base.yaml`) add values, don't rewrite existing definitions. |
| Prompts | Built from `configs/categories/generated/*.yaml` + `configs/attributes/*.yaml` + `configs/guidelines/*.yaml` + `attribute_values.yaml`. Only YAML values reach the prompt, never `#` comments. Inference builds the prompt live; training bakes it in at `prepare_dataset`. |
| Test set | Frozen in `configs/split_registry.json` (4 images per subcategory). New images only fill missing test slots. Commit the registry after every `prepare_dataset`. |
| Review exports | Always `--export-dir outputs/reviewed`, never straight into `data/raw/labels`. |
| Push before asking for review | Project knowledge syncs from GitHub; unpushed edits are invisible. |
| Decoding / image size | Greedy decoding; `image.thumbnail((896, 896))` in both training and inference. |

Bash reminders: variables are `${ct}`, not `{$ct}`. Long jobs: `nohup script.sh > /dev/null 2>&1 &`, and keep Windows from sleeping.

---

## 1. Taxonomy change (Windows, INFLOWE app)

Edit the hierarchy / attribute values in the app as usual, then:

```powershell
# prune / merge values of one attribute
python manage.py enh_populate_data_client --client base --only-prune --attribute <attribute>
python manage.py merge_taxonomy_values --attribute <attribute> --map old=new
python manage.py enh_populate_data_client --client base --only-prune --prune-delete --attribute <attribute>

# fingerprint (note the value: WSL must show the same)
python -m inflowe_taxonomy fingerprint --client base
```

Commit the taxonomy sub-repo, then bump it in the app repo:

```powershell
cd E:\INFLOWE\inflowe-taxonomy
git commit -am "<what changed>"
git push
cd E:\INFLOWE
git add inflowe-taxonomy
git commit -m "Bump taxonomy: <what changed> [skip tests]"
git push
```

Populate the databases with the new taxonomy (local, then Heroku once the deploy has finished).
`--expect-fingerprint` refuses to run if the instance does not have the taxonomy you expect.
Populate only creates/updates, never deletes, so a full populate is always safe:

```powershell
# local (Windows)
python manage.py enh_populate_data_client --client base --expect-fingerprint <hash>                  # all attributes
python manage.py enh_populate_data_client --client base --attribute category_hierarchy --expect-fingerprint <hash>
python manage.py enh_populate_data_client --client base --only-prune                                  # orphan report
python manage.py enh_populate_data_client --client base --only-prune --prune-delete                   # delete unreferenced orphans

# Heroku: quote the whole command and name the app
heroku run "python manage.py enh_populate_data_client --client base --expect-fingerprint <hash>" -a inflowe-base
heroku run "python manage.py enh_populate_data_client --client base --only-prune" -a inflowe-base
heroku run "python manage.py enh_populate_data_client --client base --only-prune --prune-delete" -a inflowe-base
```

Other options: `--attribute <a>` (one attribute), `--dry-run`, `--list-attributes`, `--list-clients`,
`--clear-attribute <a>` / `--clear` (destructive: clears data before populating).
Orphans are reported as `KEEP` (still used by items, never deleted), `UNUSED` (report run) or `DELETE` (`--prune-delete` run).
Retag the `KEEP` items first (`merge_taxonomy_values --attribute <a> --map old=new`), then run the prune with `--prune-delete` again.
The prune follows `inflowe-taxonomy/data/registry.yaml`: an attribute without `model:` or with `prune: false` is skipped.

### 1b. Removed or renamed subcategory/category

`category_hierarchy` is pruned too (registry: `model: SubCategory`; the prune checks subcategories by
(category slug, subcategory slug) and categories by slug). Order matters:

1. **Populate first** (`--attribute category_hierarchy --expect-fingerprint <hash>`), so the new subcategories exist.
2. **Retag items** that still use the old subcategory (Django shell, below). Look up the subcategory by
   **name within its category**: the same name exists in several categories (e.g. `jacket`).
3. **Prune report**, then **`--prune-delete`**, with `--attribute category_hierarchy`. Accepted leftovers stay as `KEEP`
   (currently `flat-shoes-men/slipper`, until the men's taxonomy is reviewed).

Retag script: write it to a file and run it via `exec`. PowerShell here-strings piped into
`manage.py shell` don't run `for` blocks, and `Set-Content -Encoding utf8` adds a BOM, so use `-Encoding ascii`
(or read with `encoding='utf-8-sig'`):

```python
# retag_<what>.py
from items.models import SubCategory as S, Category as C, Item as I
targets = {
    "<item-slug>": ("<category-slug>", "<subcategory name>"),
}
for slug, (cs, sname) in targets.items():
    it = I.objects.get(slug=slug)
    cat = C.objects.get(slug=cs)
    sub = S.objects.get(category=cat, name=sname)
    it.category.clear(); it.category.add(cat)
    it.subcategory.clear(); it.subcategory.add(sub)
    print(slug, "->", cs, "/", sname)
```

```powershell
# local
python manage.py shell -c "exec(open('retag_<what>.py', encoding='utf-8-sig').read())"
# Heroku: the file is not on the dyno, so open a shell and paste the code
#   (press Enter on an empty line after the for block)
heroku run "python manage.py shell" -a inflowe-base
```

`SubCategory.DoesNotExist` during the retag = that instance was not populated with the new taxonomy yet (step 1).

Naming rules:
- subcategory names in ASCII (accents only in the UI label, e.g. `minaudiere`);
- spaces between words (`flap shoulder bag`), hyphens only in fixed names (`t-shirt`) and catch-alls (`other-*`);
- one catch-all per group is fine (`other flat shoes`, `other shoulder bags`, `flap shoulder bag`) but it must stay small and come last in the decision order.

---

## 2. Align WSL with the new taxonomy

```bash
cd ~/projects/local-vlm-project
git submodule update --remote inflowe-taxonomy
python -m inflowe_taxonomy fingerprint --client base      # same value as on Windows
git add inflowe-taxonomy
git commit -m "Bump taxonomy: <what changed>"
git push

python scripts/generate_category_configs.py               # rewrites configs/categories/generated/
python scripts/category_config.py shoes_women | head -60  # taxonomy line shows the change
```

Generator warnings about RTW subcategory names shared by several categories are expected.

After a taxonomy change, also:
- add/remove the guideline entry and decision-order step (section 3);
- relabel images of removed/renamed subcategories (section 5, `--subcategories "<old name>"`);
- update `taxonomy_fingerprint` in `configs/adapters.yaml` when the adapter is retrained.

---

## 3. Guidelines and value definitions

Files: `configs/guidelines/<macro>.yaml` (`_decision_order_<gender>` block + one entry per subcategory), `configs/guidelines/attribute_values.yaml` (definitions of attribute values), `configs/attributes/<macro>.yaml` (vocabularies).

Rules of thumb:
- Decide the family by overall shape first, split it with one or two visible details.
- Every check must be visible in a product photo (no ratios to measure, no "fits in one hand", no "functional vs decorative").
- The decision order must be MECE: each item stops at exactly one step; catch-alls come last.
- Don't repeat a rule in many entries; the prompt budget is about 6,100 tokens (75% of 8,192).

Check what the model will actually see:

```bash
python scripts/category_config.py bags_women | grep -n -i "<word>"   # rule present / old wording gone
python scripts/category_config.py bags_women | sed -n '60,100p'       # read a block
git add configs/guidelines && git commit -m "<macro>: <rule change>" && git push
```

A guideline change needs `prepare_dataset` + retraining to affect the model (section 7).

---

## 4. New batch of images

```bash
# 1. images (renamed, flat folder)
#    data/raw/images/<config>_batchNN/        e.g. bags_women_batch05

# 2. duplicates within the batch and against all earlier batches
python scripts/check_duplicates.py data/raw/images/bags_women_batch05
#    hash < 40 and colour < 5 = same photo; drop the NEW copy if it matches an older batch
#    options: --only-new, --max-dist 60, --max-colour, --max-aspect

# 3. pre-label with the current adapter
python scripts/infer.py --images-dir data/raw/images/bags_women_batch05 \
  --category bags_women --adapter models/qwen3vl_v13/checkpoint-267 \
  --output outputs/drafts/bags_women_batch05_drafts.json

# 4. review (as carefully as test data: new images can enter the test set)
python scripts/review_labels.py --drafts outputs/drafts/bags_women_batch05_drafts.json \
  --category bags_women --export-dir outputs/reviewed --batch-name batch05
#    reopen later with --resume

# 5. remove duplicates from the export if needed (clean_duplicates.py: set path + drop list),
#    then copy without overwriting
f=data/raw/labels/bags_women_batch05.json
[ -e "$f" ] && echo "EXISTS - check first" || cp outputs/reviewed/bags_women_batch05.json "$f"

# 6. validate the batch alone
python scripts/taxonomy_utils.py --labels "$f" --config bags_women --strict
```

Add it to the adapter's manifest (and to an experiment manifest if used):

```yaml
  - name: bags_women_batch05
    labels: data/raw/labels/bags_women_batch05.json
    images_dir: data/raw/images/bags_women_batch05
```

A new batch is never merged: `review_queue merge` only updates images already in a batch.

Loop over two folders:
```bash
for ct in shoes bags; do
  python scripts/infer.py --images-dir data/raw/images/${ct}_women_batch05 \
    --category ${ct}_women --adapter models/qwen3vl_v13/checkpoint-267 \
    --output outputs/drafts/${ct}_women_batch05_drafts.json
done
```

---

## 5. Correct existing labels (gallery → queue → review → merge)

```bash
# 1. look at a whole group
python scripts/attribute_gallery.py --config bags_women --group subcategory \
  --subcategories "satchel bag,messenger,flap shoulder bag"
python scripts/attribute_gallery.py --config shoes_women --group heel_height+sole_type
python scripts/attribute_gallery.py --config bags_women --group model_details
#    output: outputs/gallery/<config>_<group>.html  (open: cmd.exe /c start <file>)
#    click images, "Copy selection" -> comma-separated names

# 2. build a queue (current labels, copied from the batch files)
python scripts/review_queue.py build --name <queue> --force --images "<pasted names>"
#    alternatives:  --subcategories "a,b"   --attr field=v1,v2   --rules
#                   --predictions outputs/<run>.json --compare subcategory

# 3. review
python scripts/review_labels.py --drafts outputs/queues/<queue>_<config>.json \
  --category <config> --export-dir outputs/reviewed --batch-name <queue>

# 4. merge back into the batch files (backups are made automatically)
python scripts/review_queue.py merge --exported outputs/reviewed/<config>_<queue>.json --dry-run
python scripts/review_queue.py merge --exported outputs/reviewed/<config>_<queue>.json
git add data/raw/labels && git commit -m "labels: <what>" && git push
```

Skipping an image in the review tool without editing it leaves its label unchanged.
Change a label only when it breaks the rules, never toward the model's answer.

Useful checks:

```bash
# images per subcategory in a manifest
python -c "
import json,yaml,collections
m=yaml.safe_load(open('configs/dataset_manifest_shoes_bags.yaml')); c=collections.Counter()
for b in m['batches']:
    for x in json.load(open(b['labels'])):
        t=x.get('tags') or {}; c[(x.get('category_config'),t.get('subcategory'))]+=1
for (cfg,s),n in sorted(c.items(), key=lambda kv: kv[1]): print(f'{n:4d}  {cfg:12s} {s}')"
# target: about 16 per subcategory (4 test + about 12 train)

# cross-batch duplicate filenames
python scripts/dedupe_batches.py            # --apply removes later copies
```

---

## 6. Build the dataset

```bash
python scripts/preflight_retrain.py --manifest configs/dataset_manifest_shoes_bags.yaml \
  --model-config configs/model_shoes_bags.yaml
#   BLOCKERS must be empty. Expected notes: values with no definition, filename-prefix
#   mismatches (filenames are only first guesses), prompt sizes.
#   Heel contradictions skip ANY_HEEL_SUBS (espadrille, clog).

python scripts/prepare_dataset.py --manifest configs/dataset_manifest_shoes_bags.yaml --strict
#   writes <output_dir>/{train,val,test}.parquet, test_manifest.csv, test_labels.json
#   (--resplit rebuilds the frozen test set: normally never)

git add configs/split_registry.json configs/ data/raw/labels \
  && git commit -m "<adapter> dataset: <what changed>" && git push
```

Raising `splits.test_per_subcategory` in the manifest only adds test images from images not yet in the registry.

---

## 7. Train

In `configs/model_<adapter>.yaml`:

```yaml
training:
  output_dir: "models/qwen3vl_vNN"     # new folder every run (preflight blocks an existing one)
  num_train_epochs: 3                   # production; 2 for quick checks
  save_total_limit: 3                   # keep every epoch's checkpoint
  eval_strategy: "epoch"
data:
  train_path: "data/splits/train.parquet"
  val_path:   "data/splits/val.parquet"
  test_path:  "data/splits/test.parquet"
```

```bash
python scripts/train_qwen3vl.py --config configs/model_shoes_bags.yaml
#   ~712 images x 3 epochs = 267 steps, ~6 h on the RTX 3090
#   checkpoints: models/qwen3vl_vNN/checkpoint-<steps>; final = last epoch
```

Overnight: train, then inference with the previous adapter and every new checkpoint (`scripts/run_vNN.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=outputs/logs/run_$(date +%Y%m%d_%H%M).log; mkdir -p outputs/logs
exec > >(tee -a "$LOG") 2>&1
NEW=qwen3vl_vNN; PREV=qwen3vl_vMM/checkpoint-267; CFG=configs/model_shoes_bags.yaml
TEST=data/splits/test_manifest.csv

echo "== $(date) training $NEW"
python scripts/train_qwen3vl.py --config "$CFG"

adapters=("$PREV")
for ck in $(ls -d models/$NEW/checkpoint-* | sort -t- -k2 -n); do adapters+=("${ck#models/}"); done
for ad in "${adapters[@]}"; do
  tag=$(echo "$ad" | tr '/' '_')
  echo "== $(date) inference $ad"
  python scripts/infer.py --manifest "$TEST" --adapter "models/$ad" --output "outputs/test_$tag.json"
done
echo "== $(date) done"
```

```bash
chmod +x scripts/run_vNN.sh
nohup scripts/run_vNN.sh > /dev/null 2>&1 &
tail -f outputs/logs/run_*.log          # Ctrl+C stops watching, not the run
```

---

## 8. Inference and evaluation

```bash
# inference on the test set (one adapter)
python scripts/infer.py --manifest data/splits/test_manifest.csv \
  --adapter models/qwen3vl_vNN/checkpoint-267 --output outputs/test_qwen3vl_vNN_checkpoint-267.json

# score all runs of a round; the previous adapter is the baseline and goes first
P=rNN                                                   # report prefix for this round
BASE=qwen3vl_vMM_checkpoint-267
tags="$BASE $(ls outputs/test_qwen3vl_vNN_checkpoint-*.json | sed 's#outputs/test_##; s#\.json##' | sort -t- -k2 -n)"
for tag in $tags; do
  python scripts/eval_heldout.py --predictions outputs/test_$tag.json \
    --ground-truth data/splits/test_labels.json \
    --out-report outputs/reports/${P}_$tag.json \
    --errors-csv outputs/reports/errors_${P}_$tag.csv \
    $( [ $tag != $BASE ] && echo --compare outputs/reports/${P}_$BASE.json )
done
```

`eval_heldout.py` options: `--target-per-subcategory N` (coverage), `--value-fields a,b`, `--values-csv FILE`, `--min-side N` / `--max-side N` with `--manifest`.

Labels changed since the last round? Rescore the previous adapter on the new labels (its predictions are still valid if the test images are the same; rerun its inference if the test set grew).

Reading the report:
- about 200 test images: 1 image = 0.5 points, ±3 points is noise; read the fixed/broken lists.
- choose the checkpoint on subcategory accuracy + macro F1 + key attributes; on a tie take the earlier one.

List the errors by our label:
```bash
python -c "
import json
r=json.load(open('outputs/reports/${P}_qwen3vl_vNN_checkpoint-267.json'))
for i,v in sorted(r['per_image'].items(), key=lambda kv: kv[1]['gt_subcategory']):
    if not v['ok']: print(f\"{v['gt_subcategory']:18s} -> {v['pred_subcategory']:18s} {i}\")"
```
Sort each error into: label wrong (fix, section 5) / rule unclear (section 3) / model gap (more images, section 4).

---

## 9. Fast experiment (one macrocategory, ~2–2.5 h)

```bash
# configs/experiments/dataset_manifest_bags.yaml : only bags_* batches, output_dir: data/splits_bags
# configs/experiments/model_bags_exp.yaml        : output_dir models/qwen3vl_bags_expN, 2 epochs,
#                                                  data paths -> data/splits_bags/...
python scripts/prepare_dataset.py --manifest configs/experiments/dataset_manifest_bags.yaml --strict
python scripts/train_qwen3vl.py --config configs/experiments/model_bags_exp.yaml

# test only the affected group
python -c "
import csv,json
grp={'satchel bag','messenger','flap shoulder bag','other shoulder bags'}
gt={x['image']:(x.get('tags') or {}).get('subcategory') for x in json.load(open('data/splits_bags/test_labels.json'))}
rows=list(csv.DictReader(open('data/splits_bags/test_manifest.csv')))
keep=[r for r in rows if gt.get(r['image_path'].split('/')[-1]) in grp]
w=csv.DictWriter(open('data/splits_bags/test_group.csv','w',newline=''),fieldnames=rows[0].keys()); w.writeheader(); w.writerows(keep)
print(len(keep),'images')"
python scripts/infer.py --manifest data/splits_bags/test_group.csv \
  --adapter models/qwen3vl_bags_expN/final --output outputs/bags_expN_group.json
python scripts/eval_heldout.py --predictions outputs/bags_expN_group.json \
  --ground-truth data/splits_bags/test_labels.json --errors-csv outputs/reports/errors_bags_expN.csv
#   "no prediction" for images outside the group is expected
```
The split registry is shared, so the test images are the same as in the full run. Finish with one full joint run.

---

## 10. Promote and freeze an adapter

```bash
# 1. record it in configs/adapters.yaml (path, version, report, manifest, model_config,
#    taxonomy_fingerprint, scores, git_tag)
# 2. snapshot everything that shaped it
git add -A && git commit -m "<adapter> vN frozen: <checkpoint>" && git push
git tag -a <adapter>-vN -m "Adapter <path>; report <report>" && git push origin <adapter>-vN
```

Model weights (`models/`) are not in git: keep the promoted checkpoint, plus the previous one as fallback, and copy them outside the project.

Working with a tag:

| Want | Command |
|---|---|
| see a file as frozen | `git show shoes-bags-v1:configs/guidelines/bags.yaml` |
| what changed since | `git diff shoes-bags-v1 -- configs/` |
| restore one file | `git restore --source shoes-bags-v1 -- <path>` |
| separate full copy to retrain/debug | `git worktree add ../vlm_shoes_bags_v1 shoes-bags-v1` |
| list tags | `git tag -l` ; on GitHub: `git ls-remote --tags origin` |

`git checkout <tag> -- configs/` overwrites every file under `configs/` with the frozen version (uncommitted edits are lost); prefer the commands above.

---

## 11. Export to the app

**1. Build the import file (WSL)**

```bash
python scripts/export_for_app.py \
  --input outputs/reviewed/bags_women_ext01.json \
  --images-dir /mnt/e/vlm_images/bags_ext01 \
  --meta configs/app-exports/bags_ext01.yaml \
  --client base \
  --adapter qwen3vl_v13_ep3 \
  --out exports/bags_ext01_import.json
```

Input: a `review_labels` export (reviewed), or raw `infer.py` output with `--allow-unreviewed` (always imported unpublished).
Gates (any failure = nothing written): training-gate validation (same as `prepare_dataset --strict`), app vocabulary of `--client`, whitelisted metadata columns, image files present, no duplicate slugs.
Options: `--drop-invalid` (exclude failing items instead of aborting), `--allow-warnings`, `--category <config>` (items without `category_config`), `--no-auto-title`, `--force` (overwrite `--out`), `--data <taxonomy dir>`.

Meta YAML (`configs/app-exports/<name>.yaml`):
```yaml
defaults:                 # whitelisted importer columns, applied to every item
  brand: Gucci
  reference_codes: [GUCCI_BAGS_02]
  season: spring
  year: "2019"
  published: false        # check in the app, then publish
  gender: women           # optional assertions: must match the category config
  macrocategory: bags
per_image_csv: path.csv   # optional: key column `slug` or `image` + whitelisted columns (override defaults)
```
Allowed columns: brand, reference_codes, published, season, year, collection, people, people_group, icon, icon_group, style, show_exit, article, title, product_purpose, line, decade (+ gender, macrocategory as checks).
Output: `{"_meta": {client, taxonomy_fingerprint, source_sha256, adapter, ...}, "items": [...]}`.

**2. Local import (Windows, Django)**

```powershell
python manage.py import_content_s3 --items exports\bags_ext01_import.json --format json --images-dir <folder with the images> --dry-run
python manage.py import_content_s3 --items exports\bags_ext01_import.json --format json --images-dir <folder with the images>
```
`--format json` implies `--strict-refs` and checks the taxonomy fingerprint (`--ignore-fingerprint` to override; better: align the taxonomy).
Other options: `--batch-size 100`, `--limit N`, `--start N`, `--quiet`, `--relative-ids`.
Writes next to the items file: `<items>_media.txt` (files for the S3 sync) and `<items>_heroku.json` (same rows with the stored image names).

**3. Sync media to S3 (local only, never from Heroku), then import on Heroku**

```powershell
python manage.py enh_sync_media --file-list exports\bags_ext01_import_media.txt
heroku run "python manage.py import_content_s3 --items exports/bags_ext01_import_heroku.json --format json --images-dir dummy" -a inflowe-base
```
`--images-dir dummy` makes Heroku reference the files already on S3 (`items/full/<stored name>`).

---

## 12. Housekeeping

```bash
# Windows Zone.Identifier files (created when copying downloads via Explorer)
find . -name "*:Zone.Identifier" -not -path "./.git/*" -print0 | xargs -0 git rm --cached --ignore-unmatch -q
find . -name "*:Zone.Identifier" -not -path "./.git/*" -delete
grep -qxF '*:Zone.Identifier' .gitignore || echo '*:Zone.Identifier' >> .gitignore
#   copy files with `cp /mnt/c/Users/<you>/Downloads/... .` inside WSL to avoid them

# open a file in the Windows default app
cmd.exe /c start outputs/gallery/<file>.html
```

Folder map:

| Path | Content |
|---|---|
| `configs/adapters.yaml` | adapter registry |
| `configs/dataset_manifest_<adapter>.yaml`, `configs/model_<adapter>.yaml` | per-adapter data + training config |
| `configs/experiments/`, `configs/legacy/` | experiment / old configs |
| `configs/guidelines/` | prompt rules and value definitions |
| `configs/categories/generated/` | generated from the taxonomy (don't edit) |
| `configs/split_registry.json` | frozen test set |
| `data/raw/images/<batch>/`, `data/raw/labels/<batch>.json` | images and ground truth |
| `data/splits*/` | built by `prepare_dataset` |
| `outputs/drafts/`, `outputs/queues/`, `outputs/reviewed/` | pre-labels, review queues, review exports |
| `outputs/reports/`, `outputs/gallery/`, `outputs/logs/` | eval reports, galleries, run logs |
| `models/qwen3vl_vNN/checkpoint-*` | trained adapters (not in git) |

---

## 13. Diagnostic tools (occasional)

| Script | Use |
|---|---|
| `scripts/audit_dataset.py --manifest configs/dataset_manifest_shoes_bags.yaml [--predictions outputs/<run>.json]` | Missing images on disk, missing descriptions/tags, per-batch subcategory counts, duplicate paths across batches, which split each image landed in; with `--predictions`, label vs prediction per image. Reads `data/splits/*.parquet`. Always pass `--manifest` (its default is the old name). |
| `scripts/extract_vocabularies.py` | Prints colour, material (flat and by group), embellishment, pattern and style lists from `inflowe-taxonomy/data/base/*.yaml`, formatted to paste into `configs/categories/_base.yaml` after a vocabulary change. Paths overridable with `--colours`, `--materials`, `--material-groups`, `--embellishments`, `--patterns`, `--styles`. |
| `scripts/find_predicted_values.py` | Ad-hoc: lists test images where the model predicted a given value (e.g. `saffiano`) next to the label's value. Edit the predictions file, the field and the value at the top before running. |
| `scripts/check_duplicates.py <dir>` | Near-duplicate images (section 4). |
| `scripts/dedupe_batches.py [--apply]` | Same filename in several batches. |
| `scripts/clean_duplicates.py` | Drops listed images from one labels file (set path + drop list inside; makes a timestamped backup). |
| `scripts/measure_sequence.py` | Token length of training samples (image + prompt + answer) vs `max_seq_length`. |
| `scripts/migrate_*.py`, `scripts/reconcile_taxonomy.py` | One-off migrations / taxonomy reconciliation used in earlier rounds; read the script header before reusing. |
