#!/bin/sh
set -x

if [ ! -d OpenSubtitles ]; then
    wget -O en.tar.gz 'http://opus.nlpl.eu/download.php?f=OpenSubtitles/en.tar.gz'
    tar xf en.tar.gz
    rm en.tar.gz
fi

[ ! -f glove.6B.zip ] && wget http://nlp.stanford.edu/data/glove.6B.zip
[ ! -f glove.6B.100d.txt ] && unzip glove.6B.zip glove.6B.100d.txt

