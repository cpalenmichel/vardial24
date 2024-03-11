from argparse import ArgumentParser
from collections import defaultdict, Counter


def collapse_labels():
    parser = ArgumentParser()
    parser.add_argument("inpath")
    parser.add_argument("outpath")
    parser.add_argument("--threshold", default=0.3, type=float)
    args = parser.parse_args()
    out_lines = defaultdict(list)
    with open(args.inpath, 'r', encoding='utf8') as infile:
        for line in infile:
            fields = line.split('\t')
            label = fields[0]
            seg_idx = fields[-1]
            for l in label.split(','):
                out_lines[int(seg_idx)].append(label)

    with open(args.outpath, 'w', encoding='utf8') as outfile:
        for seg_idx in sorted(out_lines):
            print(seg_idx)
            cntr = Counter(out_lines[seg_idx])
            label_proportions = {l: cntr[l] / float(sum(cntr.values())) for l in cntr}
            labels = [l for l, val in label_proportions.items() if val > args.threshold]
            print(','.join(labels), file=outfile)


if __name__ == "__main__":
    collapse_labels()