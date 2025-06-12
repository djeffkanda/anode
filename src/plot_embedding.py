import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datamanager.dataset import *
import argparse
import os

from src.utils.utils import get_X_from_loader

#
def load_dataset(dataset_name, dataset_path, pct=1.0, contamination_rate=0, holdout=.4):
    """
    Load and prepare dataset using only a percentage of the data

    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to the dataset file
        pct: Percentage of data to use (between 0 and 1)
    """
    cls_name = f'{dataset_name}Dataset'
    dataset_cls = globals()[cls_name]
    dataset = dataset_cls(path=dataset_path, pct=pct)

    train_ldr, test_ldr, neg_val_ldr, val_ldr = dataset.loaders(batch_size=1024,
                                                                seed=42,
                                                                contamination_rate=contamination_rate,
                                                                validation_ratio=0,
                                                                holdout=holdout)

    # Get all data and labels
    X, y = get_X_from_loader(train_ldr)
    # y = train_ldr.dataset.y

    # If pct < 1, take a random subset of the data
    # if pct < 1.0:
    #     n_samples = int(len(X) * pct)
    #     indices = np.random.choice(len(X), n_samples, replace=False)
    #     X = X[indices]
    #     y = y[indices]

    # Standardize the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, dataset_name


def create_embeddings(X, y, dataset_name, random_state=42):
    """Create UMAP and t-SNE embeddings"""
    # UMAP
    umap = UMAP( n_neighbors=15, min_dist=0.1) #random_state=random_state,
    umap_embedding = umap.fit_transform(X)

    # t-SNE
    tsne = TSNE( learning_rate='auto', init='random', perplexity=30) #random_state=random_state,
    tsne_embedding = tsne.fit_transform(X)

    return umap_embedding, tsne_embedding


def plot_embeddings(umap_embedding, tsne_embedding, y, dataset_name, output_dir='plots'):
    """Plot UMAP and t-SNE embeddings separately with legends"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create color map for binary classification
    colors = np.where(y == 0, 'green', 'red')
    
    # Create and save UMAP plot
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_embedding[y == 0][:, 0], umap_embedding[y == 0][:, 1],
               c='blue', alpha=0.6, label='Benign')
    plt.scatter(umap_embedding[y == 1][:, 0], umap_embedding[y == 1][:, 1],
               c='red', alpha=0.6, label='Abnormal')
    # plt.title(f'UMAP embedding - {dataset_name}')
    plt.xlabel('UMAP 1', fontdict={'fontsize': 14})
    plt.ylabel('UMAP 2', fontdict={'fontsize': 14})
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'umap_embedding_{dataset_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create and save t-SNE plot
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_embedding[y == 0][:, 0], tsne_embedding[y == 0][:, 1],
               c='blue', alpha=0.6, label='Benign')
    plt.scatter(tsne_embedding[y == 1][:, 0], tsne_embedding[y == 1][:, 1],
               c='red', alpha=0.6, label='Abnormal')
    # plt.title(f't-SNE embedding - {dataset_name}')
    plt.xlabel('t-SNE 1', fontdict={'fontsize': 14})
    plt.ylabel('t-SNE 2', fontdict={'fontsize': 14})
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tsne_embedding_{dataset_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate UMAP and t-SNE visualizations')
    parser.add_argument('--dataset_root', type=str, required=False,
                        help='Root directory containing datasets')
    parser.add_argument('--pct', type=float, default=.1,
                        help='Percentage of data to use (between 0 and 1)')
    parser.add_argument('--output_dir', type=str, default='emb_plots',
                        help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Dataset paths
    # path_to_datasets_map = {
    #     # "Thyroid": f"{args.dataset_root}/thyroid.mat",
    #     "NSLKDD": f"/home/local/USHERBROOKE/nkad2101/projects/anado/data/nsl/3_minified/NSL-KDD_minified.npz",
    #     # "USBIDS": f"{args.dataset_root}/usb-ids.npz",
    #     # "IDS2018": f"{args.dataset_root}/ids2018.npz"
    #     "Kitsune": "/home/local/USHERBROOKE/nkad2101/projects/anado/raw_data/kitsune_/processed/kitsune.gzip",
    #     "IOT2023" : "/home/local/USHERBROOKE/nkad2101/projects/anado/raw_data/CICIoT2023/processed/ciciot2023_40.gzip",
    # }

    path_to_datasets_map = {
        # "Arrhythmia": f"{args.dataset_root}/arrhythmia.npz",
        "KDD10": f"/home/local/USHERBROOKE/nkad2101/projects/anado/data/kdd10/3_minified/KDD10percent_minified.npz",
        # "Thyroid": f"{args.dataset_root}/thyroid.mat",
        # "NSLKDD": f"/home/local/USHERBROOKE/nkad2101/projects/anado/data/nsl/3_minified/NSL-KDD_minified.npz",
        # "USBIDS": f"{args.dataset_root}/usb-ids.npz",
        # "IDS2018": f"{args.dataset_root}/ids2018.npz"
        # "Kitsune": "/home/local/USHERBROOKE/nkad2101/projects/anado/raw_data/kitsune_/processed/kitsune.gzip",
        # "IOT2023" : "/home/local/USHERBROOKE/nkad2101/projects/anado/raw_data/CICIoT2023/processed/ciciot2023_40.gzip",
        "CIC18Imp": "/home/local/USHERBROOKE/nkad2101/projects/anado/data/cic18imp0.09.gzip"
    }

    # Process each dataset
    for dataset_name, dataset_path in path_to_datasets_map.items():
        print(f"Processing {dataset_name} (using {args.pct * 100}% of data)...")
        try:
            # Load and prepare data
            X, y, name = load_dataset(dataset_name, dataset_path, args.pct, contamination_rate=.12)

            print(f"Dataset shape: {X.shape}")
            print(f"Number of classes: {len(np.unique(y))}")

            # Create embeddings
            umap_embedding, tsne_embedding = create_embeddings(X, y, name, args.seed)

            # Plot and save visualizations
            plot_embeddings(umap_embedding, tsne_embedding, y, name, args.output_dir)

            print(f"Successfully processed {dataset_name}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            print("-" * 50)


if __name__ == "__main__":
    main()