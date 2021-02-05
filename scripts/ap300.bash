#!/bin/sh -e
type isort autopep8 autoflake pylint black || pip install -U isort autopep8 autoflake pylint black
n=300
isort $*
# use black in n=120 mode it mangles dictionaries otherwise
# black --skip-string-normalization -l $n $*
autopep8 --max-line-length=$n -i $*
autoflake -i $*
pylint --errors-only $* || :
# pylint is noisy, you can ignore a lot of stuff but it sometimes helps catch things
