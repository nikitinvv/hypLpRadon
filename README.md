HypLpRadon


OVERVIEW

Evaluation of the hyperbolic Radon transform by using the log-polar-based method, for details see:

[1] Nikitin, Viktor, Fredrik Andersson, Marcus Carlsson, and Anton Duchkov. "Fast hyperbolic Radon transform represented as
convolutions in log-polar coordinates." 
https://www.sciencedirect.com/science/article/pii/S0098300417304806


INSTALLATION

conda create -n hypradon -c conda-forge scikit-build swig notebook matplotlib

conda activate hypradon

cd hypLpRadon

python setup.py install


TESTS

See the jupyter notebook "hypRadon.ipynb"

jupyter notebook hypRadon.ipynb
