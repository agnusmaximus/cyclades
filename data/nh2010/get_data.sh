wget -qO- http://www.cise.ufl.edu/research/sparse/MM/DIMACS10/nh2010.tar.gz | bsdtar -xvf- && python extract_sparse_matrix.py nh2010/nh2010.mtx nh2010/nh2010.data
