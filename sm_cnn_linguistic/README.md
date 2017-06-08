# Linguistic Perspective Experiments

cd to `Castor` and download stanford core nlp tools from [here](http://nlp.stanford.edu/software/corenlp.shtml):
```
cd Castor/sm_cnn_linguistic
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
unzip stanford-corenlp-full-2016-10-31.zip
```

Current PyTorch version uses all CPUs in a machine.
Set the following environments to limit the number of CPUs to 4.
```
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```