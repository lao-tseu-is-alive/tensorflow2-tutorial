#!/bin/bash
for i in *.jpg; do mv "$i" "${i%.jpg}.jpeg"; done
ls *.jpeg | echo `wc -l` " jpeg files"