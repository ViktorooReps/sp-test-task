wget -P ./glove-embs/ "http://nlp.stanford.edu/data/glove.6B.zip"
unzip ./glove-embs/glove.6B.zip -d glove-embs/
rm ./glove-embs/glove.6B.zip

python extract.py