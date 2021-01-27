#!/bin/bash
for i in `ls -1 *.jpeg`; do basename $i .jpeg; done|sort -n