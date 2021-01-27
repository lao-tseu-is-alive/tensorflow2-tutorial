#!/bin/bash
cd godata_resized
for i in `ls  ../godata/correspondance/*.jpeg`; do convert $i -resize 256 correspondance/`basename $i`; done
for i in `ls  ../godata/facturation/*.jpeg`; do convert $i -resize 256 facturation/`basename $i`; done
for i in `ls  ../godata/plan/*.jpeg`; do convert $i -resize 256 plan/`basename $i`; done
for i in `ls  ../godata/plan_situation/*.jpeg`; do convert $i -resize 256 plan_situation/`basename $i`; done
for i in `ls  ../godata/plan_projet/*.jpeg`; do convert $i -resize 256 plan_projet/`basename $i`; done
for i in `ls  ../godata/photo/*.jpeg`; do convert $i -resize 256 photo/`basename $i`; done
