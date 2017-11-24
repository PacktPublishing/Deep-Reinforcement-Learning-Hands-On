#!/usr/bin/env bash
FNAME=ch08-small-quotes.tgz
rm -f $FNAME
wget https://www.dropbox.com/s/z2qt7tmylcp18f7/ch08-small-quotes.tgz && tar xvf $FNAME && rm -f $FNAME
