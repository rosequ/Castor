#!/bin/sh
if [[ $# -gt 0 ]]; then
    for file in $(find $1 -name '*.toks'); do
        #basic tags
        java -mx4g -cp "./*:" edu.stanford.nlp.pipeline.StanfordCoreNLP  \
        -tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -ssplit.eolonly true -tokenize.whitespace true \
        -annotators tokenize,ssplit,pos,lemma,ner \
        -ner.model "../stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz" \
        -file $file -ner.useSUTime false -outputFormat JSON \
        -outputDirectory $(dirname $file) -outputExtension ".basic.json"
    done
fi


        java -mx4g -cp "./*:" edu.stanford.nlp.pipeline.StanfordCoreNLP  \
        -ner.model "../stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz" \
        -tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -ssplit.eolonly true \
        -annotators tokenize,ssplit,pos,lemma -file $file -ner.useSUTime false -outputFormat JSON \
        -outputDirectory $(dirname $file) -outputExtension ".basic.json"

java -mx700m -cp "./stanford-ner.jar:./lib/*" edu.stanford.nlp.ie.crf.CRFClassifier \
        -loadClassifier ./classifiers/english.all.3class.distsim.crf.ser.gz \
        -tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -tokenizerOptions "tokenizeNLs=true" \
        -textFile ${file} > ${file}.basic.ner

        echo $file
        java -mx4g -cp "./*:" edu.stanford.nlp.pipeline.StanfordCoreNLP  \
        -tokenizerFactory edu.stanford.nlp.process.WhitespaceTokenizer -ssplit.eolonly true \
        -annotators tokenize,ssplit,pos,lemma,ner -file $file -ner.useSUTime false -outputFormat JSON \
        -outputDirectory $(dirname $file) -outputExtension ".full.json"