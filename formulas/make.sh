#!/bin/bash

[ -d ch01 ] || mkdir ch01
# create full doc
pdflatex -jobname=ch01/ch01 ch01.tex
# create images
pdflatex "\def\ispreview{1} \input{ch01.tex}"
convert -density 300 ch01.pdf -quality 90 ch01/ch01.png
rm ch01/*.{aux,log}
