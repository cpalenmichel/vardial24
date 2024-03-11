from argparse import ArgumentParser

import spacy


def split_bcms():
    parser = ArgumentParser()
    parser.add_argument("inpath")
    parser.add_argument("outpath")
    parser.add_argument("--is-test", action='store_true')
    args = parser.parse_args()

    nlp = spacy.load("hr_core_news_sm")

    new_sents = []
    with open(args.inpath, 'r', encoding='utf8') as infile:
        for orig_line_num, line in enumerate(infile):
            print(f"Completed {orig_line_num} lines")
            fields = line.strip().split('\t')
            if args.is_test:
                text = fields[0]
            else:
                label = fields[0]
                text = fields[1]
            doc = nlp(text)
            for sent in doc.sents:
                if len(sent) < 5:
                    continue
                if args.is_test:
                    new_line = f"{sent.text}\t{orig_line_num}"
                else:
                    new_line = f"{label}\t{sent.text}\t{orig_line_num}"
                new_sents.append(new_line)

    with open(args.outpath, 'w', encoding='utf8') as outfile:
        for new_line in new_sents:
            print(new_line, file=outfile)


if __name__ == "__main__":
    split_bcms()