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

    parser.add_argument('--dataset_name')
    parser.add_argument('--num_additional_points')
    parser.add_argument('--radius')
    
    return parser

def _multiply(dataset_name: str, radius: float, num_additional_points: int):
    
    data_filename = f"data_{dataset_name}.dat"
    colors_filename = f"data_{dataset_name}_colors.dat"
    
    data = np.fromfile(data_filename, dtype='float32')
    colors = np.fromfile(colors_filename, dtype='float32')
    
    data = np.reshape(data, (-1,3))
    colors = np.reshape(colors, (-1,4))
    
    data = np.tile(data,(num_additional_points+1, 1))
    colors = np.tile(colors,(num_additional_points+1, 1))
        
    noise = np.random.uniform(low=-radius, high=radius, size=data.shape)
    data = data + noise
    
    noise = np.random.uniform(low=-0.1, high=0.0, size=colors.shape)
    colors = colors + noise
    
    colors.astype('float32').tofile(f'data_{dataset_name}_multiplied_colors.dat')
    data.astype('float32').tofile(f'data_{dataset_name}_multiplied.dat')

if __name__ == "__main__":
    parser = _create_parser()
    
    args: dict = parser.parse_args()
    
    _multiply(dataset_name=str(args.dataset_name), num_additional_points=int(args.num_additional_points), radius=float(args.radius))

