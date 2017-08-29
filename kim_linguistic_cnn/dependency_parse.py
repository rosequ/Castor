import argparse
import json
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
properties={'annotators': 'depparse', 'outputFormat': 'json'}

def parse(fname):
    save_output = ''
    with open(fname, 'r') as fhandle, open(fname + '.deps.json', 'w') as whandle:
        for line in fhandle:
            sentence = line.strip().split(' ', 1)
            output = nlp.annotate(sentence[1], properties)
            save_output += json.dumps(output['sentences'][0]) + ",\n"
        whandle.write("[" + save_output[:-2:] + "]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='dependnecy parsing and reordering of sentence tokens')
    ap.add_argument("input")
    args = ap.parse_args()

    parse(args.input)
