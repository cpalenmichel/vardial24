"""
Overly simplistic data augmentation strategy.
Splice two halves of sentences together to generate new pairs.
Use halves from same label.
"""

import random
from argparse import ArgumentParser
from collections import defaultdict

random.seed(42)


def naive_augment():
    parser = ArgumentParser()
    parser.add_argument("inpath")
    parser.add_argument("outpath")
    parser.add_argument("--augment-ratio", type=float, default=1.0)
    args = parser.parse_args()

    texts_by_label = defaultdict(list)
    start_chunks_by_label = defaultdict(list)
    end_chunks_by_label = defaultdict(list)
    with open(args.inpath, "r", encoding="utf8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            label = fields[0]
            text = fields[1]
            texts_by_label[label].append(text)

            # Naive split for pseudo tokens, better if used real tokenizer
            tokens = text.split()
            split_point = int(len(tokens) / 2)
            first_half = tokens[:split_point]
            second_half = tokens[split_point:]
            start_chunks_by_label[label].append(" ".join(first_half))
            end_chunks_by_label[label].append(" ".join(second_half))

    with open(args.outpath, "w", encoding="utf8") as outfile:
        for label in texts_by_label:
            for n in range(int((len(texts_by_label[label]) * args.augment_ratio))):
                new_text = (
                    random.choice(start_chunks_by_label[label])
                    + " "
                    + random.choice(end_chunks_by_label[label])
                )
                print(f"{label}\t{new_text}", file=outfile)


if __name__ == "__main__":
    naive_augment()
