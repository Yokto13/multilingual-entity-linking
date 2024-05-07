from collections import defaultdict
import json
from pathlib import Path
import pandas as pd
import glob

from tqdm import tqdm

output_path = Path("outputs")

mentions_counts = lambda lang: str(output_path / f"damuel_mentions_{lang}.json/part*.json")

langs = ["ar", "de", "en", "fa", "ja", "sr", "ta", "tr"]

language_count = {}

for lang in langs:
    with open(output_path / f"language_counts_{lang}.json", "r") as f:
        language_count[lang] = json.load(f)[lang]

print(language_count)

language_qid_counts = defaultdict(dict)

for lang in tqdm(language_count):
    json_files = glob.glob(mentions_counts(lang))
    dfs = [pd.read_json(file, lines=True) for file in json_files]
    df = pd.concat(dfs, ignore_index=True)

    language_qid_counts[lang] = dict(zip(df['qid'], df['count']))

qids = set(qid for lang_qids in language_qid_counts.values() for qid in lang_qids.keys())

qid_loc_pair = {}
qid_loc_pair_debug = {}
print("QIDs:", len(qids))
for qid in tqdm(qids):
    qid_options = []
    for lang, qid_count in language_qid_counts.items():
        if qid in qid_count:
            qid_options.append((lang, qid_count[qid], language_count[lang]))
        else:
            qid_options.append((lang, 0, language_count[lang]))
    qid_options.sort(key=lambda x: (x[1], x[2]), reverse=True)
    qid_loc_pair[qid] = qid_options[0][0]
    if qid_options[0][1] != 0:
        qid_loc_pair_debug[qid] = qid_options
print(qid_options)


# print("QID options:", qid_loc_pair)
print("QID loc pair debug:", list(qid_loc_pair_debug.items())[:10000])
print(len(qid_loc_pair), len(qids), len(qid_loc_pair_debug))


with open(output_path / "qid_loc_pair.json", "w") as f:
    json.dump(qid_loc_pair, f)