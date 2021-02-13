# Mutual Information

<img src="img/coots.jpg" alt="reeeeeedme" width="400px">

Estimating [differential entropy](https://en.wikipedia.org/wiki/Differential_entropy) and mutual information.

Non-parametric computation of differential entropy and mutual-information.
Originally adapted by G Varoquaux in a [gist](https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429)
for code created by R Brette, itself from several papers (see in the code).
These computations rely on nearest-neighbor statistics.
* [Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy of a random vector. Probl. Inf. Transm. 23, 95-101](http://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=ppi&paperid=797&option_lang=eng)
* [Evans, D. 2008 A computationally efficient estimator for mutual information, Proc. R. Soc. A 464 (2093), 1203-1215](https://royalsocietypublishing.org/doi/10.1098/rspa.2007.0196)
* [Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual information. Phys Rev E 69(6 Pt 2):066138](https://arxiv.org/abs/cond-mat/0305641)
* [F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures for Continuous Random Variables. Advances in Neural Information Processing Systems 21 (NIPS). Vancouver (Canada), December.](https://papers.nips.cc/paper/2008/hash/ccb0989662211f61edae2e26d58ea92f-Abstract.html)

See `Makefile` for example ops.

See https://pypi.org/project/mutual-info

Do not pin packages for now. Let's surf latest and find out when things break.

## Install

    python setup.py install

    or

    pip install pypi

## Develop install

    python setup.py develop

## Tests

    make test

## TODO

* incorporate fixes from @thismartian (see thismartian branch)
* test shift invariance of entropy (not multiplicative)
* test triangle inequality of mutual information
* test symmetry of mutual information
* test scale and shift invariance of mutual information (any smooth invertible transformation)

