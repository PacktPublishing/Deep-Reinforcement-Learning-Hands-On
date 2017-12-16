#!/bin/bash

[ -d ch01 ] || mkdir ch01
pdflatex ch01.tex
convert -density 300 ch01.pdf -quality 90 ch01/ch01.png
