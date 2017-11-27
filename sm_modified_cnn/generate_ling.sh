#!/bin/sh
if [[ $# -gt 0 ]]; then
    for file in $(find $1 -name '*.toks'); do
        #basic tags
        java -mx4g -cp "./*:" edu.stanford.nlp.pipeline.StanfordCoreNLP  \
        -tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -ssplit.eolonly true -tokenize.whitespace true \
        -annotators tokenize,ssplit,pos,lemma,depparse,ner \
        -file $file -ner.useSUTime false -outputFormat JSON \
        -outputDirectory $(dirname $file) -outputExtension ".json"
    done
fi
