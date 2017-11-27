#!/bin/sh
mkdir -p data
#python parse_json.py --input ../../data/TrecQA/
#python overlap_features.py --dir ../../data/TrecQA/

CURRENT_DIR=$(pwd)
#cd ../../data/TrecQA
#todo: rename ner_feats_basic to ner_feats
#cd raw-dev/; paste ${CURRENT_DIR}/data/backup.dev.tsv ner_feats > $CURRENT_DIR/data/trecqa.dev.tsv; cd ..
#cd raw-test/; paste ${CURRENT_DIR}/data/backup.test.tsv ner_feats > $CURRENT_DIR/data/trecqa.test.tsv; cd ..
#cd train-all/; paste ${CURRENT_DIR}/data/backup.train.tsv ner_feats > $CURRENT_DIR/data/trecqa.train.tsv; cd ..
#cd $CURRENT_DIR

#ToDo: add linguistic features for WikiQA dataset
python parse_ling.py --input ../../data/cleanQA/
python overlap_features.py --dir ../../data/cleanQA/

cd ../../data/cleanQA
cd clean-dev/; paste id.txt sim.txt a.toks b.toks overlap_feats.txt dependency_feats  idf_feats.txt  > $CURRENT_DIR/data/cleanqa.dev.tsv; cd ..
cd clean-test/; paste id.txt sim.txt a.toks b.toks overlap_feats.txt dependency_feats idf_feats.txt  > $CURRENT_DIR/data/cleanqa.test.tsv; cd ..
cd train-all/; paste id.txt sim.txt a.toks b.toks overlap_feats.txt dependency_feats idf_feats.txt  > $CURRENT_DIR/data/cleanqa.train.tsv; cd ..
cd $CURRENT_DIR
