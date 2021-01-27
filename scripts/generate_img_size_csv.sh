#!/bin/bash
cd godata
identify -format "%w,%h\n" facturation/*.jpeg |sort -n > facturation_sizes.csv
identify -format "%w,%h\n" plan/*.jpeg |sort -n > plan_sizes.csv 
identify -format "%w,%h\n" photo/*.jpeg |sort -n > photo_sizes.csv 
identify -format "%w,%h\n" plan_situation/*.jpeg |sort -n > plan_situation_sizes.csv 
identify -format "%w,%h\n" plan_projet/*.jpeg |sort -n > plan_projet_sizes.csv 
identify -format "%w,%h\n" correspondance/*.jpeg |sort -n > correspondance_sizes.csv 
