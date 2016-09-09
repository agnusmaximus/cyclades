wget -qO- http://konect.uni-koblenz.de/downloads/tsv/dblp-author.tar.bz2 | tar xvz && python ../nh2010/extract_sparse_matrix.py dblp-author/out.dblp-author dblp-author/dblp.data
