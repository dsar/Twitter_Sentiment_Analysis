#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat $1 $2 | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../data/glove/vocab.txt
