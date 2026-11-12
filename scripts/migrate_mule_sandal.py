"""
One-off migration for taxonomy v2 (2026-11): remove `mule sandal` from sandals-women.

  * mule sandal -> flat mule   (flat-shoes-women)   when heel_height is flat-heel / low-heel
  * mule sandal -> heeled mule (heeled-shoes-women) when heel_height is mid-heel / high-heel
                                                     or heel_type is kitten-heel
  * "mule sandal" in the description becomes "mule"
  * prints a REVIEW list: migrated samples that may really be strappy sandals
    (ankle strap, buckle closure, "strap" in the description, missing heel height)
  * prints MARY JANE FLAT candidates (other flat shoes / d'orsay flat / ballerina whose
    description mentions an instep strap or Mary Jane). Nothing is changed for those.

Plan only by default; --apply rewrites the files, keeping a .bak_mulesandal copy of each.

Usage:
    python scripts/migrate_mule_sandal.py
    python scripts/migrate_mule_sandal.py --apply
    python scripts/migrate_mule_sandal.py --labels "data/raw/labels/*.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLAT = {"flat-heel", "low-heel"}
HEELED = {"mid-heel", "high-heel"}
MULE_SANDAL_RE = re.compile(r"\bmule[ -]sandal\b", re.IGNORECASE)
MARY_JANE_RE = re.compile(r"mary[ -]jane|instep strap|strap across the instep", re.IGNORECASE)
MARY_JANE_FROM = {"other flat shoes", "d'orsay flat", "ballerina"}


def samples_of(data):
    """Label files are a list of samples; tolerate {"samples": [...]} too."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return data["samples"]
    return []


def fix_description(text: str) -> str:
    def repl(m: re.Match) -> str:
        return "Mule" if m.group(0)[0].isupper() else "mule"
    return MULE_SANDAL_RE.sub(repl, text or "")


def target_for(tags: dict) -> tuple[str, str, str | None]:
    """(category, subcategory, review_reason or None)."""
    hh, ht = tags.get("heel_height"), tags.get("heel_type")
    if ht == "kitten-heel" or hh in HEELED:
        return "heeled-shoes-women", "heeled mule", None
    if hh in FLAT:
        return "flat-shoes-women", "flat mule", None
    if ht and ht != "none":
        return "heeled-shoes-women", "heeled mule", f"no heel_height (heel_type {ht}): guessed heeled"
    return "flat-shoes-women", "flat mule", "no heel_height: guessed flat"


def review_reasons(sample: dict) -> list[str]:
    tags, desc = sample.get("tags") or {}, sample.get("description") or ""
    reasons = []
    if "ankle-strap" in (tags.get("model_details") or []):
        reasons.append("ankle-strap: not backless? -> sandal")
    if tags.get("closure") == "buckle":
        reasons.append("buckle closure")
    if re.search(r"\bstraps?\b", desc, re.IGNORECASE):
        reasons.append("'strap' in description: thin straps -> sandal")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(PROJECT_ROOT / "data/raw/labels/*.json"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(args.labels) if f.endswith(".json"))
    review, mary_jane, total = [], [], 0

    for f in files:
        path = Path(f)
        data = json.loads(path.read_text(encoding="utf-8"))
        changed = 0
        for s in samples_of(data):
            tags = s.get("tags") or {}
            sub = tags.get("subcategory")
            if sub in MARY_JANE_FROM and MARY_JANE_RE.search(s.get("description") or ""):
                mary_jane.append((path.name, s.get("image"), sub))
            if sub != "mule sandal":
                continue
            cat, new_sub, guess = target_for(tags)
            reasons = review_reasons(s) + ([guess] if guess else [])
            print(f"  {path.name:40s} {s.get('image'):35s} {tags.get('heel_height')!s:10s} -> {new_sub}")
            if reasons:
                review.append((path.name, s.get("image"), new_sub, "; ".join(reasons)))
            tags["category"], tags["subcategory"] = cat, new_sub
            s["description"] = fix_description(s.get("description"))
            changed += 1
        total += changed
        if changed and args.apply:
            shutil.copy2(path, path.with_name(path.name + ".bak_mulesandal"))
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{total} mule sandal sample(s) {'migrated' if args.apply else 'to migrate (plan only, use --apply)'}")
    print(f"\nREVIEW ({len(review)}): may be strappy sandals or need a heel check")
    for r in review:
        print("  " + " | ".join(r))
    print(f"\nMARY JANE FLAT candidates ({len(mary_jane)}): check and relabel in the review UI")
    for m in mary_jane:
        print("  " + " | ".join(m))


if __name__ == "__main__":
    main()
