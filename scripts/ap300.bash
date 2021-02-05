#!/bin/sh -e
type isort autopep8 autoflake pylint black || pip install -U isort autopep8 autoflake pylint black
n=300
isort $*
# use black in n=120 mode it mangles dictionaries otherwise
# black --skip-string-normalization -l $n $*
autopep8 --max-line-length=$n -i $*
# do not write with autoflake just warn
autoflake $*
pylint --errors-only $*
