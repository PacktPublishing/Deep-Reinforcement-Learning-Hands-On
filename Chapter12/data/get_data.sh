#!/bin/sh
set -x

[ ! -f cornell_movie_dialogs_corpus.zip ] && wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
[ ! -d "cornell movie-dialogs corpus" ] && unzip -x cornell_movie_dialogs_corpus.zip && rm -rf __MACOSX
[ ! -d cornell ] && mv "cornell movie-dialogs corpus" cornell
