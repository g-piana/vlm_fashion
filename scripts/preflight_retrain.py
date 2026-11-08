"""
Pre-flight check before retraining: is everything that feeds the prompts and the
labels in sync?   Run from the project root:

    python scripts/preflight_retrain.py            # exit code 1 if any BLOCKER
    python scripts/preflight_retrain.py --skip-regen

Checks
  1. generated configs == what generate_category_configs.py produces now (stale-config guard)
  2. guidelines/<macro>.yaml keys vs subcategories (orphans are silently dropped from the prompt)
  3. guidelines/attribute_values.yaml keys vs vocabularies (same silent drop)
  4. labels validated with --strict (out-of-vocabulary values are warnings, not errors, by default)
  5. label sanity: filename prefix vs subcategory, heeled/flat vs heel_height
  6. prompt size per config vs max_seq_length
"""
from __future__ import annotations

import argparse, json, re, subprocess, sys, tempfile, unicodedata
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from category_config import (build_system_prompt, list_configs, load_category_config,  # noqa: E402
                             load_guidelines, load_value_definitions)

blockers: list[str] = []
notes: list[str] = []


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def load(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def check_generated(skip: bool) -> None:
    if skip:
        return
    gen = ROOT / "configs/categories/generated"
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, str(HERE / "generate_category_configs.py"),
                            "--output-dir", tmp], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            blockers.append(f"generator failed: {r.stderr[-300:]}")
            return
        for f in sorted(Path(tmp).glob("*.yaml")):
            cur = gen / f.name
            if not cur.exists():
                blockers.append(f"generated config missing in repo: {f.name}")
            elif load(cur) != load(f):
                blockers.append(f"STALE generated config (differs from taxonomy): {f.name}")
        for f in gen.glob("*.yaml"):
            if not (Path(tmp) / f.name).exists():
                blockers.append(f"generated config no longer produced by taxonomy: {f.name}")


def check_guidelines(names: list[str]) -> None:
    by_macro: dict[str, set[str]] = {}
    for n in names:
        cfg = load_category_config(n)
        m = cfg.get("macrocategory")
        if m:
            by_macro.setdefault(m, set()).update(s for subs in cfg["taxonomy"].values() for s in subs)
    for macro, subs in by_macro.items():
        g = set(load_guidelines(macro))
        if g - subs:
            blockers.append(f"guidelines/{macro}.yaml keys not in taxonomy (dropped from prompt): {sorted(g - subs)}")
        if g and subs - g:
            notes.append(f"guidelines/{macro}.yaml has no entry for: {sorted(subs - g)}")
        if not g:
            notes.append(f"no guidelines file for macro '{macro}'")

    # attribute_values.yaml is shared by all macros, so a definition is stale only if NO config
    # has the value in that attribute's vocabulary. Extra definitions are ignored by the prompt.
    defs = load_value_definitions()

    def vocab(attrs: dict, attr: str, depth: int = 0) -> set[str]:
        spec = attrs.get(attr)
        v = spec.get("values") if isinstance(spec, dict) else None
        if isinstance(v, str) and v.startswith("same_as:") and depth < 5:
            return vocab(attrs, v.split(":", 1)[1].strip(), depth + 1)
        return {str(x) for x in v} if isinstance(v, list) else set()

    union: dict[str, set[str]] = {}
    per_cfg = {n: load_category_config(n).get("attributes", {}) for n in list_configs()}
    for attrs in per_cfg.values():
        for attr in defs:
            union.setdefault(attr, set()).update(vocab(attrs, attr))
    for attr, vals in defs.items():
        stale = sorted(set(vals) - union.get(attr, set()))
        if stale:
            notes.append(f"attribute_values.yaml '{attr}': defined but in no config's vocabulary: {stale}")
    for n in names:
        for attr, vals in defs.items():
            missing = sorted(vocab(per_cfg[n], attr) - set(vals))
            if missing:
                notes.append(f"{n}: '{attr}' vocabulary values with no definition: {missing}")


def check_labels(manifest: Path) -> None:
    r = subprocess.run([sys.executable, str(HERE / "taxonomy_utils.py"), "--manifest", str(manifest), "--strict"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        blockers.append("label validation --strict FAILED (see output below)")
        print(r.stdout[-3000:], r.stderr[-1500:])

    mism, contra = Counter(), []
    for b in load(manifest).get("batches", []):
        p = ROOT / b["labels"]
        if not p.exists():
            blockers.append(f"labels file missing: {p}")
            continue
        for s in json.loads(p.read_text(encoding="utf-8")):
            t = s.get("tags")
            if not isinstance(t, dict):
                continue
            img = s.get("image", "")
            if "__" in img:
                pre, sub = img.split("__")[0], slug(t.get("subcategory", ""))
                if pre != sub:
                    mism[(b["name"], pre, t.get("subcategory"))] += 1
            cat, hh = str(t.get("category", "")), t.get("heel_height")
            if ("heeled" in cat and hh == "flat-heel") or ("flat-shoes" in cat and hh in ("mid-heel", "high-heel")):
                contra.append((b["name"], img, cat, hh))
    for (batch, pre, sub), n in mism.most_common(15):
        notes.append(f"{batch}: {n} file(s) named '{pre}__*' labelled subcategory '{sub}'")
    for c in contra[:15]:
        notes.append(f"heel contradiction: {c}")
    if contra:
        notes.append(f"... {len(contra)} heel_height/category contradictions in total")


def check_prompts(names: list[str], max_len: int) -> None:
    for n in names:
        for it in ("product", "look"):
            try:
                p = build_system_prompt(n, it)
            except Exception as e:
                blockers.append(f"prompt build failed {n}|{it}: {e}")
                continue
            est = len(p) / 3.5
            if it == "product" and est > max_len * 0.75:
                blockers.append(f"{n}: system prompt ~{est:.0f} tokens, > 75% of max_seq_length={max_len}")
            elif it == "product":
                notes.append(f"{n}: system prompt ~{est:.0f} of {max_len} tokens (chars/3.5 estimate; "
                             f"image + answer need roughly 2k more)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "configs/dataset_manifest.yaml")
    ap.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    ap.add_argument("--skip-regen", action="store_true")
    a = ap.parse_args()

    manifest = load(a.manifest)
    used = sorted({json.loads((ROOT / b["labels"]).read_text(encoding="utf-8"))[0].get("category_config", "shoes_women")
                   for b in manifest.get("batches", []) if (ROOT / b["labels"]).exists()})
    all_cfgs = [c for c in list_configs() if c != "look"]
    max_len = load(a.model_config)["model"]["max_seq_length"]

    check_generated(a.skip_regen)
    check_guidelines(used or all_cfgs)
    check_labels(a.manifest)
    check_prompts(used or all_cfgs, max_len)

    out_dir = load(a.model_config)["training"]["output_dir"]
    if (ROOT / out_dir).exists():
        blockers.append(f"training output_dir already exists and would be overwritten: {out_dir}")

    print("\n== NOTES ==");    [print("  -", n) for n in notes]
    print("\n== BLOCKERS =="); [print("  x", b) for b in blockers] or None
    print("\nREADY" if not blockers else f"\n{len(blockers)} blocker(s): fix before retraining")
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
