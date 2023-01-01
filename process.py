import argparse
import anndata
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np
import logging

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')    

    parser.add_argument('--dataset-name')
    
    return parser

def _process(dataset_name: str, num_pca_components: int, num_pca_iterations: int, num_clusters: int):
    
    anndata_file = anndata.read_h5ad(f'anndata/{dataset_name}.h5ad')
    X = anndata_file.X
    svd = TruncatedSVD(n_components=num_pca_components, n_iter=num_pca_iterations, random_state=42)
    
    X_cvded = svd.fit_transform(X)
    
    X_embedded = TSNE(n_components=3, 
                      learning_rate='auto',  
                      init='random', 
                      perplexity=3, 
                      verbose=3, 
                      n_iter=250).fit_transform(X_cvded)
    
    X_embedded_scaled = X_embedded * 5
    
    X_embedded_scaled.astype('float32').tofile(f"data_{dataset_name}.dat")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(X_embedded)
    
    palette = np.random.uniform(size=(num_clusters, 4))
    
    num_points = X_embedded_scaled.shape[0]
    
    colors = np.zeros(dtype='float32', shape=(num_points, 4))
    for i, cluster_id in enumerate(kmeans.labels_): # type: ignore 
        colors[i][:] = palette[cluster_id]
        
    colors.astype('float32').tofile(f'data_{dataset_name}_colors.dat')
    


if __name__ == "__main__":
    parser = _create_parser()
    
    args: dict = parser.parse_args()
    
    _process(dataset_name=str(args.dataset_name), num_pca_components=10, num_pca_iterations=5, num_clusters=5)
    
    print("Hello, World!")
