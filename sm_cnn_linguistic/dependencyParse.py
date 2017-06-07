from nltk.parse.stanford import StanfordDependencyParser

def dependency_parse(sentence):
    path_to_jar = 'stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar'
    path_to_models_jar = 'stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0-models.jar'
    dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

    result = dependency_parser.raw_parse(sentence)
    dep = result.__next__()

    # dependency_tagged = ""
    # for item in list(dep.triples()):
    #     dependency_tagged += "{} {} ".format(item[0][0], item[2][0])
    return list(dep.triples())

if __name__ == "__main__":
    print(dependency_parse("I like mangoes"))