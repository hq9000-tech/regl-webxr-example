import anndata
from sklearn.manifold import TSNE
anndata_file = anndata.read_h5ad('anndata/local.h5ad')
type(anndata_file.X)
X = anndata_file.X

X_embedded = TSNE(n_components=3, learning_rate='auto',  init='random', perplexity=3, verbose=3, n_iter=250).fit_transform(X)