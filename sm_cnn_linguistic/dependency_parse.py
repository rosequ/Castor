from nltk.parse.stanford import StanfordDependencyParser

import string

path_to_jar = 'stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar'
path_to_models_jar = 'stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0-models.jar'
dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

def dependency_parse(sentence):
    result = dependency_parser.raw_parse(sentence)
    dep = result.__next__()
    return list(dep.triples())

def get_dependency_reordered(sentence):
    deps = dependency_parse(sentence)
    reordered_tokens = []
    for item in deps:
        reordered_tokens.append(item[0][0])
        reordered_tokens.append(item[2][0])
    return ' '.join(reordered_tokens)

if __name__ == "__main__":
    print(dependency_parse("I like mangoes"))