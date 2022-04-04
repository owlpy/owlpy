#!/bin/bash

set -e

rm -rf data
wget -r https://owlpy.org/examples/data/ -nv -e 'robots=off' -R 'index.html' --no-parent -nH --cut-dirs=2 -P data
find data -name '*.tmp' -print0 | xargs -0 rm
