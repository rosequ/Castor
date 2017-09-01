import argparse
import re
import json
from pprint import pprint
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
properties={'annotators': 'depparse', 'outputFormat': 'json'}

def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip()

def parse(fname):
    with open(fname, 'r') as fhandle, open(fname + '.deps.json', 'w') as whandle:
        for line in fhandle:
            sentence = clean_str_sst(line).split(' ', 1)

            if len(sentence) > 1:
                output = nlp.annotate(sentence[1], properties)
                whandle.write(json.dumps(output['sentences'][0]) + "\n")

def read_json(fname):
    with open(fname) as json_file:
        for line in json_file:
            a = json.loads(line)
            pprint(a)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='dependnecy parsing and reordering of sentence tokens')
    ap.add_argument("input")
    args = ap.parse_args()

    parse(args.input)
    # read_json(args.input + '.deps.json')