from anndata import AnnData
import numpy as np
import sklearn.neighbors as sk_neighbors

from typing import Optional

NN_QUERIES_PER_CHUNK = 15000

def build_neighbor_index(
    x: np.ndarray[tuple[int, int], np.dtype[np.number]], leaf_size: int
) -> sk_neighbors.BallTree:
    return sk_neighbors.BallTree(x, leaf_size=leaf_size)

def compute_nearest_neighbors(matrix, balltree, k: int, row_start: int):
    nn_dist, nn_idx = balltree.query(matrix, k=k+1)
    nn_idx = nn_idx[:, 1:]
    nn_dist = nn_dist[:, 1:]

    i: np.ndarray[tuple[int, int], np.dtype[np.int_]] = np.repeat(
        row_start + np.arange(nn_idx.shape[0]), k
    )
    j: np.ndarray[tuple[int, int], np.dtype[np.int_]] = nn_idx.ravel().astype(int)
    return (i, j, nn_dist.ravel())

def hd_cluster(
    adata: AnnData,
    spatial_key: str = "spatial",
    key_added: Optional[str] = "hdc",
    pca_key: str = "pca",
    copy: bool = False
) -> Optional[AnnData]:
    
    pca_mat = adata.obsm[pca_key]

    use_bcs = np.arange(pca_mat.shape[0])

    # build nearest neigbor query index
    balltree = build_neighbor_index(pca_mat, balltree_leaf_size)

    use_neighbors = int(max(
        given_num_neighbors, np.round(given_neighbor_a + given_neighbor_b * np.log10(len(use_bcs)))
    ))
    num_neighbors = max(1, min(use_neighbors, len(use_bcs) - 1))

    chunks = []
    for row_start in range(0, pca_mat.shape[0], NN_QUERIES_PER_CHUNK):
        row_end = min(row_start + NN_QUERIES_PER_CHUNK, pca_mat.shape[0])

        # Write the pca submatrix to an h5 file
        submatrix_path = martian.make_path("%d_submatrix.h5" % row_start)
        cr_graphclust.save_ndarray_h5(
            pca_mat[row_start:row_end, :], submatrix_path, "submatrix"
        )

        chunks.append(
            {
                "neighbor_index": neighbor_index,
                "submatrix": submatrix_path,
                "row_start": row_start,
                "total_rows": pca_mat.shape[0],
                "k_nearest": num_neighbors,
                "use_bcs": use_bcs_path,
            }
        )
    return adata if copy else None