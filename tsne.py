import anndata
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
anndata_file = anndata.read_h5ad('anndata/organoid.h5ad')
type(anndata_file.X)
X = anndata_file.X

svd = TruncatedSVD(n_components=10, n_iter=14, random_state=42)

X_cvded = svd.fit_transform(X)

X_embedded = TSNE(n_components=3, learning_rate='auto',  init='random', perplexity=3, verbose=3, n_iter=11).fit_transform(X_cvded)
