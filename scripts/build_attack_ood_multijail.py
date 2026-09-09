"""
Builds the MultiJail covariate/multilingual Attack-OOD test set (Deng et
al. 2024, "Multilingual Jailbreak Challenges in Large Language Models").

KNOWN, ACCEPTED CONFOUND -- logged explicitly, not glossed over: unlike
every other eval condition in this project, MultiJail's underlying
English base prompts are NOT AdvBench. Checked directly against the
released `source` column: 300/315 are "anthropics", 15/315 are "openai".
This means results here mix a language-shift effect with a content-shift
effect (MultiJail's own English prompts are already a different
distribution from AdvBench, closer in spirit to the Anthropic Red Team
material already used as a semantic-OOD source elsewhere in this
project). The methodologically clean version of this test would
translate our OWN AdvBench goals into target languages ourselves, holding
content fixed. That was raised and explicitly deferred -- the user chose
to use MultiJail as released for now rather than wait on a translation-
method decision. Any interpretation of this run's results should name
this confound, not present the delta as a clean language-only effect.

Builds one flat prompt file across all 9 non-English languages released
(zh, it, vi, ar, ko, th, bn, sw, jv -- 3 high/3 medium/3 low-resource
per the paper's own framing), 315 prompts each, 2,835 total. English is
deliberately NOT included in this eval file -- it's just MultiJail's own
base content, redundant with existing semantic-OOD sources, not itself an
attack-OOD test. Metadata retains id/source/tags/language per record for
later per-language breakdown.

Source: https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs
(data/MultiJail.csv, fetched via plain git clone -- unlike jailbreak_llms,
this repo cloned normally).

Usage:
    python scripts/build_attack_ood_multijail.py
"""
import csv
import json
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_URL = "https://raw.githubusercontent.com/DAMO-NLP-SG/multilingual-safety-for-LLMs/main/data/MultiJail.csv"
OUT_PROMPTS = REPO_ROOT / "data" / "attack_ood_multijail.csv"
OUT_METADATA = REPO_ROOT / "data" / "attack_ood_multijail_metadata.json"

LANGUAGES = ["zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]  # non-English only


def main():
    local_csv = "/tmp/MultiJail.csv"
    urllib.request.urlretrieve(SOURCE_URL, local_csv)
    with open(local_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} MultiJail base rows")

    records = []
    for row in rows:
        for lang in LANGUAGES:
            text = row[lang].replace("\n", " ").strip()
            records.append({
                "id": row["id"], "source": row["source"], "tags": row["tags"],
                "language": lang, "prompt": text,
            })

    with open(OUT_PROMPTS, "w", encoding="utf-8") as f:
        f.write("\n".join(r["prompt"] for r in records) + "\n")
    json.dump({"records": records}, open(OUT_METADATA, "w"), indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} prompts ({len(rows)} base x {len(LANGUAGES)} languages) -> {OUT_PROMPTS}")
    print(f"Wrote metadata -> {OUT_METADATA}")


if __name__ == "__main__":
    main()
