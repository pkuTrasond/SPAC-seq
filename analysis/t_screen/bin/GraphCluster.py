import numpy as np
import scipy.sparse as sp_sparse
from scipy.cluster.hierarchy import linkage
import louvain as louvain
from typing import Optional
import pandas as pd
import subprocess

import sklearn.neighbors as sk_neighbors
from anndata import AnnData

import sys
sys.path.append('/home/wpy/stereoseq/20240502-SPACseq/bin')

from fast_utils import(
    compute_sseq_params_o3,
    sseq_differential_expression_o3
)

LOUVAIN_DEFAULT_SEED = 0

LOUVAIN_BIN_PATH = "/home/wpy/stereoseq/20240502-SPACseq/bin/louvain"
CONVERT_BIN_PATH = "/home/wpy/stereoseq/20240502-SPACseq/bin/convert"
LOUVAIN_PATH = "/home/wpy/stereoseq/20240502-SPACseq/bin/louvain_tmp"

def hdc(
        adata: AnnData,
        pca_key: str = "X_pca",
        leaf_size: int = 40,
        num_neighbors: int = 1,
        neighbor_a: int = -230,
        neighbor_b: int = 120,
        random_seed: int = LOUVAIN_DEFAULT_SEED,
        result_key: str = "hdc"
) -> Optional[AnnData]:
    # load X data from adata
    expr_matrix = adata.X.T

    # read pca information from adata
    pca_matrix = adata.obsm[pca_key]

    # build nearest neighbor query index
    balltree = sk_neighbors.BallTree(pca_matrix, leaf_size=leaf_size)

    # compute nearest neighbors
    num_cells = len(adata.obs_names)
    use_neighbors = int(max(num_neighbors, np.round(neighbor_a + neighbor_b * np.log10(num_cells))))
    n_neighbors = max(1, min(use_neighbors, num_cells - 1))

    nn_dist, nn_idx = balltree.query(pca_matrix, k=n_neighbors+1)

    nn_idx = nn_idx[:, 1:]
    nn_dist = nn_dist[:, 1:]

    i: np.ndarray[tuple[int, int], np.dtype[np.int_]] = np.repeat(
            np.arange(nn_idx.shape[0]), n_neighbors
        )
    j: np.ndarray[tuple[int, int], np.dtype[np.int_]] = nn_idx.ravel().astype(int)
    nn_matrix = nn_dist.ravel()
    
    nn = sp_sparse.coo_matrix((nn_matrix, (i, j)), shape=(pca_matrix.shape[0], pca_matrix.shape[0]))

    # pipe matrix to run louvain
    with subprocess.Popen([f"{CONVERT_BIN_PATH}", "-i", "-", "-o", f"{LOUVAIN_PATH}/matrix"], stdin=subprocess.PIPE, text=True, stdout=False) as proc:
        assert proc.stdin is not None
        try:
            for i, j in zip(nn.row, nn.col):
                proc.stdin.write(f"{i}\t{j}\n")
        finally:
            proc.stdin.close()
            proc.wait()

    # run louvain
    with open(f"{LOUVAIN_PATH}/out", 'w') as f:
        subprocess.call([f"{LOUVAIN_BIN_PATH}", f"{LOUVAIN_PATH}/matrix", "-q", "0", "-l", "-1", "-s", f"{random_seed}"], stdout=f)
        f.close()

    # load louvain results
    labels: np.ndarray[int, np.dtype[np.int64]] = np.zeros(len(adata.obs_names), dtype=np.int64)
    seen_idx = set()
    with open(f"{LOUVAIN_PATH}/out", 'r') as f:
        for line in f:
            used_bc_idx, cluster = map(int, line.strip().split(" "))

            if used_bc_idx in seen_idx: continue
            seen_idx.add(used_bc_idx)
            labels[used_bc_idx] = 1 + cluster
        f.close()

    pca_df = pd.DataFrame(pca_matrix)
    checked_cluster_pairs = set()

    while True:
        if len(np.bincount(labels)) == 1:
            break

        pca_df["cluster"] = labels
        medoids = pca_df.groupby("cluster").apply(lambda x: x.median(axis=0)).to_numpy()[:, :-1]

        hc = linkage(medoids, "complete")
        max_label = np.max(labels)

        any_merged = False
        for step in range(hc.shape[0]):
            if hc[step, 0] <= max_label and hc[step, 1] <= max_label:
                leaf0, leaf1 = hc[step, 0], hc[step, 1]
                group0, group1 = np.flatnonzero(labels == leaf0), np.flatnonzero(labels == leaf1)

                set0, set1 = frozenset(group0), frozenset(group1)
                cluster_pair = tuple(sorted([set0, set1]))

                if cluster_pair in checked_cluster_pairs: continue
                checked_cluster_pairs.add(cluster_pair)

                sub_matrix = expr_matrix[:, np.concatenate((group0, group1))]
                sub_matrix = sp_sparse.csc_matrix(sub_matrix)
                if sub_matrix.has_sorted_indices:
                    sub_matrix = sub_matrix.sorted_indices()
                #print(sub_matrix)
                params = compute_sseq_params_o3(sub_matrix.astype(np.int32), 0.995)

                group0_submat = np.arange(len(group0))
                group1_submat = np.arange(len(group0), len(group0) + len(group1))

                diff_exp_results = sseq_differential_expression_o3(sub_matrix.astype(np.int32), group0_submat, group1_submat, params, 900)

                n_de_genes = np.sum(diff_exp_results["adjusted_p_values"] < 0.05)
                if n_de_genes == 0:
                    labels[labels == leaf1] = leaf0
                    labels[labels > leaf1] = labels[labels > leaf1] - 1

                    any_merged = True
                    break
            
        if not any_merged:
            break

    labels += 1
    order: np.ndarray[int, np.dtype[np._IntType]] = np.argsort(np.argsort(-np.bincount(labels)))
    labels = 1 + order[labels]

    # output result to adata
    adata.obs[result_key] = pd.Categorical(labels.astype(str))
    