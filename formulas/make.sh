#!/bin/bash

set -x

for ch in ch01 ch04 ch05 ch06 ch07 ch09 ch10 ch12 ch14 ch15 ch16 ch18; do
    [ -d $ch ] && continue
    mkdir $ch
#    [ -d $ch ] || mkdir $ch
    # create full doc
    pdflatex -jobname=$ch/$ch $ch.tex
    pdflatex "\def\ispreview{1} \input{$ch.tex}"

    # create EPS files
    pages=`pdfinfo $ch.pdf | grep Pages | sed 's/  */ /g' | cut -d ' ' -f 2-`
    for page in `seq $pages`; do
        pdftops -f $page -l $page -eps $ch.pdf $ch/$ch-`printf '%03d' $page`.eps
    done

    # create images
#    convert -density 300 $ch.pdf -quality 90 $ch/$ch.png
    rm $ch/*.{aux,log}
done

./clean.sh
