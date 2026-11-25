"""Recreate the ordered Perception Test validation cohort with seed 0."""
import argparse
import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"


def select(annotations=None):
    if annotations is None:
        with gzip.open(DATA / "selection_pool.tsv.gz", "rt", encoding="utf-8") as stream:
            population = [(r["video_id"], int(r["question_id"]), int(r["target_count"]), r["area"])
                          for r in csv.DictReader(stream, delimiter="\t")]
    else:
        videos = json.loads(annotations.read_text(encoding="utf-8"))
        population = [(video, int(q["id"]), len(q["answers"]), q["area"])
                      for video, entry in videos.items() for q in entry["grounded_question"]]
    strata = defaultdict(list)
    for video, question, count, area in sorted(population):
        strata[("1" if count == 1 else "2+", area)].append((video, question))
    keys = sorted(strata)
    exact = {k: 200 * len(strata[k]) / len(population) for k in keys}
    allocation = {k: int(exact[k]) for k in keys}
    remainder = 200 - sum(allocation.values())
    for k in sorted(keys, key=lambda k: exact[k] - allocation[k], reverse=True)[:remainder]:
        allocation[k] += 1
    rng = random.Random(0)
    selected = []
    for k in keys:
        selected.extend(rng.sample(strata[k], allocation[k]))
    return "".join(f"{video}\t{qid}\n" for video, qid in sorted(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, help="official grounded_question_valid.json")
    parser.add_argument("--output", type=Path, help="write the verified TSV here")
    args = parser.parse_args()
    result = select(args.annotations).encode("utf-8")
    if result != (DATA / "perception_val200.tsv").read_bytes():
        raise ValueError("Selected queries differ from the archived ordered list")
    if args.output:
        args.output.write_bytes(result)
    print("Matched the archived 200 queries from 190 videos, in order.")


if __name__ == "__main__":
    main()
