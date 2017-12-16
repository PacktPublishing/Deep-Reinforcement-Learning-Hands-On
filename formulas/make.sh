#!/bin/bash

for ch in ch01 ch04; do
    [ -d $ch ] || mkdir $ch
    # create full doc
    pdflatex -jobname=$ch/$ch $ch.tex
    # create images
    pdflatex "\def\ispreview{1} \input{$ch.tex}"
    convert -density 300 $ch.pdf -quality 90 $ch/$ch.png
    rm $ch/*.{aux,log}
done
