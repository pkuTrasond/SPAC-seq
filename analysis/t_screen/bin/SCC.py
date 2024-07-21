from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, Literal, Iterable
import numpy as np
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData
from scipy import sparse as sp
import pandas as pd

def correct_hnsw_neighbors(knn_hn: np.ndarray, distances_hn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Corrects the indices and corresponding distances obtained from a hnswlib by manually adding self neighbors.

    Args:
        knn_hn: Array containing the k-NN indices obtained from the hnswlib.
        distances_hn: Array containing the distances corresponding to the k-NN indices obtained from the HNSW index.

    Returns:
        A tuple containing the corrected indices and distances.
    """
    mask = knn_hn[:, 0] == np.arange(knn_hn.shape[0])
    target_indices = np.where(mask)[0]

    def roll(arr, value=0):
        arr = np.roll(arr, 1, axis=0)
        arr[0] = value
        return arr

    knn_corrected = [knn_hn[i] if i in target_indices else roll(knn_hn[i], i) for i in range(knn_hn.shape[0])]
    distances_corrected = [
        distances_hn[i] if i in target_indices else roll(distances_hn[i]) for i in range(distances_hn.shape[0])
    ]
    return np.vstack(knn_corrected), np.vstack(distances_corrected)

def k_nearest_neighbors(
    X: np.ndarray,
    k: int,
    query_X: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    metric: Union[str, Callable] = "euclidean",
    metric_kwads: Dict[str, Any] = None,
    exclude_self: bool = True,
    knn_dim: int = 10,
    pynn_num: int = 5000,
    pynn_dim: int = 2,
    hnswlib_num: int = int(2e5),
    pynn_rand_state: int = 19491001,
    n_jobs: int = -1,
    return_nbrs: bool = False,
    **kwargs,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, NearestNeighbors]]:
    
    if method is None:
        if X.shape[0] > hnswlib_num:
            method = "hnswlib"
        elif X.shape[0] > pynn_num and X.shape[1] > pynn_dim:
            method = "pynn"
        else:
            if X.shape[1] > knn_dim:
                method = "ball_tree"
            else:
                method = "kd_tree"

    if query_X is None:
        query_X = X

    if method.lower() in ["pynn", "umap"]:
        from pynndescent import NNDescent

        nbrs = NNDescent(
            X,
            metric=metric,
            n_neighbors=k + 1,
            n_jobs=n_jobs,
            random_state=pynn_rand_state,
            **kwargs,
        )
        nbrs_idx, dists = nbrs.query(query_X, k=k + 1)
    elif method in ["ball_tree", "kd_tree"]:
        from sklearn.neighbors import NearestNeighbors

        # print("***debug X_data:", X_data)
        nbrs = NearestNeighbors(
            n_neighbors=k + 1,
            metric=metric,
            metric_params=metric_kwads,
            algorithm=method,
            n_jobs=n_jobs,
            **kwargs,
        ).fit(X)
        dists, nbrs_idx = nbrs.kneighbors(query_X)
    elif method == "hnswlib":
        try:
            import hnswlib
        except ImportError:
            raise ImportError("hnswlib is not installed, please install it first")

        space = "l2" if metric == "euclidean" else metric
        if space not in ["l2", "cosine", "ip"]:
            raise ImportError(f"hnswlib nearest neighbors with space {space} is not supported")
        nbrs = hnswlib.Index(space=space, dim=X.shape[1])
        nbrs.init_index(max_elements=X.shape[0], random_seed=pynn_rand_state, **kwargs)
        nbrs.set_num_threads(n_jobs)
        nbrs.add_items(X)
        nbrs_idx, dists = nbrs.knn_query(query_X, k=k + 1)
        if space == "l2":
            dists = np.sqrt(dists)
        nbrs_idx, dists = correct_hnsw_neighbors(nbrs_idx, dists)
    else:
        raise ImportError(f"nearest neighbor search method {method} is not supported")

    nbrs_idx = np.array(nbrs_idx)
    if exclude_self:
        nbrs_idx = nbrs_idx[:, 1:]
        dists = dists[:, 1:]
    if return_nbrs:
        return nbrs_idx, dists, nbrs, method
    return nbrs_idx, dists

def log1p_(adata: AnnData, X_data: np.ndarray) -> np.ndarray:
    """Perform log(1+x) X_data if adata.uns["pp"]["layers_norm_method"] is None.

    Args:
        adata: The AnnData that has been preprocessed.
        X_data: The data to perform log1p on.

    Returns:
        The log1p result data if "layers_norm_method" in adata is None; otherwise, X_data would be returned unchanged.
    """
    if "layers_norm_method" not in adata.uns["pp"].keys():
        return X_data
    else:
        if adata.uns["pp"]["layers_norm_method"] is None:
            if sp.issparse(X_data):
                X_data.data = np.log1p(X_data.data)
            else:
                X_data = np.log1p(X_data)

        return X_data

def fetch_X_data(adata: AnnData, genes: List, layer: str, basis: Optional[str] = None) -> Tuple:
    """Get the X data according to given parameters.

    Args:
        adata: Anndata object containing gene expression data.
        genes: List of gene names to be fetched. If None, all genes are considered.
        layer: Layer of the data to fetch.
        basis: Dimensionality reduction basis. If provided, the data is fetched from a specific embedding.

    Returns:
        A tuple containing a list of fetched gene names and the corresponding gene expression data (X data).
    """
    if basis is not None:
        return None, adata.obsm["X_" + basis]

    if genes is not None:
        genes = adata.var_names.intersection(genes).to_list()
        if len(genes) == 0:
            raise ValueError("No genes from your genes list appear in your adata object.")

    if layer is None:
        if genes is not None:
            X_data = adata[:, genes].X
        else:
            if "use_for_dynamics" not in adata.var.keys():
                X_data = adata.X
                genes = adata.var_names
            else:
                X_data = adata[:, adata.var.use_for_dynamics].X
                genes = adata.var_names[adata.var.use_for_dynamics]
    else:
        if genes is not None:
            X_data = adata[:, genes].layers[layer]
        else:
            if "use_for_dynamics" not in adata.var.keys():
                X_data = adata.layers[layer]
                genes = adata.var_names
            else:
                X_data = adata[:, adata.var.use_for_dynamics].layers[layer]
                genes = adata.var_names[adata.var.use_for_dynamics]

            X_data = log1p_(adata, X_data)

    return genes, X_data


def generate_neighbor_keys(result_prefix: str = "") -> Tuple[str, str, str]:
    """Generate neighbor keys for other functions to store/access info in adata.

    Args:
        result_prefix: The prefix for keys. Defaults to "".

    Returns:
        A tuple (conn_key, dist_key, neighbor_key) for key of connectivity matrix, distance matrix, neighbor matrix,
        respectively.
    """

    if result_prefix:
        result_prefix = result_prefix if result_prefix.endswith("_") else result_prefix + "_"
    if result_prefix is None:
        result_prefix = ""

    conn_key, dist_key, neighbor_key = (
        result_prefix + "connectivities",
        result_prefix + "distances",
        result_prefix + "neighbors",
    )
    return conn_key, dist_key, neighbor_key

def get_conn_dist_graph(knn: np.ndarray, distances: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Compute connection and distance sparse matrix.

    Args:
        knn: A matrix (n x n_neighbors) storing the indices for each node's n_neighbors nearest neighbors in knn graph.
        distances: The distances to the n_neighbors the closest points in knn graph.

    Returns:
        A tuple (distances, connectivities), where distance is the distance sparse matrix and connectivities is the
        connectivity sparse matrix.
    """

    n_obs, n_neighbors = knn.shape
    distances = sp.csr_matrix(
        (
            distances.flatten(),
            (np.repeat(np.arange(n_obs), n_neighbors), knn.flatten()),
        ),
        shape=(n_obs, n_obs),
    )
    connectivities = distances.copy()
    connectivities.data[connectivities.data > 0] = 1

    distances.eliminate_zeros()
    connectivities.eliminate_zeros()

    return distances, connectivities


def neighbors(
    adata: AnnData,
    X_data: np.ndarray = None,
    genes: Optional[List[str]] = None,
    basis: str = "pca",
    layer: Optional[str] = None,
    n_pca_components: int = 30,
    n_neighbors: int = 30,
    method: Optional[str] = None,
    metric: Union[str, Callable] = "euclidean",
    metric_kwads: Dict[str, Any] = None,
    cores: int = 1,
    seed: int = 19491001,
    result_prefix: str = "",
    **kwargs,
) -> AnnData:

    if X_data is None:
        if basis == "pca" and "X_pca" not in adata.obsm_keys():
            from ..preprocessing.pca import pca

            CM = adata.X if genes is None else adata[:, genes].X
            cm_genesums = CM.sum(axis=0)
            valid_ind = np.logical_and(np.isfinite(cm_genesums), cm_genesums != 0)
            valid_ind = np.array(valid_ind).flatten()
            CM = CM[:, valid_ind]
            adata, _, _ = pca(adata, CM, pca_key="X_pca", n_pca_components=n_pca_components, return_all=True)

            X_data = adata.obsm["X_pca"]
        else:
            genes, X_data = fetch_X_data(adata, genes, layer, basis)

    knn, distances = k_nearest_neighbors(
        X_data,
        k=n_neighbors - 1,
        method=method,
        metric=metric,
        metric_kwads=metric_kwads,
        exclude_self=False,
        pynn_rand_state=seed,
        n_jobs=cores,
        **kwargs,
    )

    conn_key, dist_key, neighbor_key = generate_neighbor_keys(result_prefix)
    adata.obsp[dist_key], adata.obsp[conn_key] = get_conn_dist_graph(knn, distances)

    adata.uns[neighbor_key] = {}
    adata.uns[neighbor_key]["indices"] = knn
    adata.uns[neighbor_key]["params"] = {
        "n_neighbors": n_neighbors,
        "method": method,
        "metric": metric,
        "n_pcs": n_pca_components,
    }

    return adata

def main_info(text: Literal[""] = ""):
    print(text)

def cluster_community_from_graph(
    graph=None,
    graph_sparse_matrix: Union[np.ndarray, sp.csr_matrix, None] = None,
    method: Literal["leiden", "louvain"] = "louvain",
    directed: bool = False,
    **kwargs
) -> Any:

    try:
        import igraph
        import leidenalg
    except ImportError:
        raise ImportError(
            "Please install networkx, igraph, leidenalg via "
            "`pip install networkx igraph leidenalg` for clustering on graph."
        )

    initial_membership = kwargs.pop("initial_membership", None)
    weights = kwargs.pop("weights", None)
    seed = kwargs.pop("seed", None)

    if graph is not None:
        # highest priority
        main_info("using graph from arg for clustering...")
    elif sp.issparse(graph_sparse_matrix):
        if directed:
            graph = igraph.Graph.Weighted_Adjacency(graph_sparse_matrix, mode="directed")
        else:
            graph = igraph.Graph.Weighted_Adjacency(graph_sparse_matrix, mode="undirected")
    else:
        raise ValueError("Expected graph inputs are invalid")

    if method == "leiden":
        # ModularityVertexPartition does not accept a resolution_parameter, instead RBConfigurationVertexPartition.
        if kwargs["resolution_parameter"] != 1:
            partition_type = leidenalg.RBConfigurationVertexPartition
        else:
            partition_type = leidenalg.ModularityVertexPartition
            kwargs.pop("resolution_parameter")

        coms = leidenalg.find_partition(
            graph, partition_type, initial_membership=initial_membership, weights=weights, seed=seed, **kwargs
        )

    elif method == "louvain":
        try:
            import louvain
        except ImportError:
            raise ImportError("Please install louvain via `pip install louvain==0.8.0` for clustering on graph.")

        coms = louvain.find_partition(
            graph,
            louvain.RBConfigurationVertexPartition,
            initial_membership=initial_membership,
            weights=weights,
            seed=seed,
            **kwargs
        )
    else:
        raise NotImplementedError("clustering algorithm not implemented yet")

    return coms

def copy_adata(adata: AnnData, logger=None) -> AnnData:
    data = adata.copy()
    return data

def cluster_community(
    adata: AnnData,
    method: Literal["leiden", "louvain"] = "leiden",
    result_key: Optional[str] = None,
    adj_matrix: Optional[Union[list, np.array, sp.csr_matrix]] = None,
    adj_matrix_key: Optional[str] = None,
    use_weight: bool = False,
    no_community_label: int = -1,
    layer: Optional[str] = None,
    obsm_key: Optional[str] = None,
    cell_subsets: Optional[List[int]] = None,
    cluster_and_subsets: Optional[Tuple[str, List[int]]] = None,
    directed: bool = True,
    copy: bool = False,
    **kwargs
) -> Optional[AnnData]:

    adata = copy_adata(adata) if copy else adata
    if (layer is not None) and (adj_matrix_key is not None):
        raise ValueError("Please supply one of adj_matrix_key and layer")
    if use_weight:
        conn_type = "distances"
    else:
        conn_type = "connectivities"

    # build adj_matrix_key
    if adj_matrix_key is None:
        if layer is None:
            if obsm_key is None:
                adj_matrix_key = conn_type
            else:
                adj_matrix_key = obsm_key + "_" + conn_type
        else:
            adj_matrix_key = layer + "_" + conn_type

    # try generating required adj_matrix according to
    # user inputs through "neighbors" interface
    if adj_matrix is None:
        main_info("accessing adj_matrix_key=%s built from args for clustering..." % (adj_matrix_key))
        if not (adj_matrix_key in adata.obsp):
            if layer is None:
                if obsm_key is None:
                    neighbors(adata)
                else:
                    X_data = adata.obsm[obsm_key]
                    neighbors(adata, X_data=X_data, result_prefix=obsm_key)
            else:
                main_info("using PCA genes for clustering based on adata.var.use_for_pca ...")
                X_data = adata[:, adata.var.use_for_pca].layers[layer]
                neighbors(adata, X_data=X_data, result_prefix=layer)

        if not (adj_matrix_key in adata.obsp):
            raise ValueError("%s does not exist in adata.obsp" % adj_matrix_key)

        graph_sparse_matrix = adata.obsp[adj_matrix_key]
    else:
        main_info("using adj_matrix from arg for clustering...")
        graph_sparse_matrix = adj_matrix

    # build result_key for storing results
    if result_key is None:
        if all((cell_subsets is None, cluster_and_subsets is None)):
            result_key = "%s" % (method) if layer is None else layer + "_" + method
        else:
            result_key = "subset_" + method if layer is None else layer + "_subset_" + method

    valid_indices = None
    if cell_subsets is not None:
        if type(cell_subsets[0]) == str:
            valid_indices = [adata.obs_names.get_loc(cur_cell) for cur_cell in cell_subsets]
        else:
            valid_indices = cell_subsets

        graph_sparse_matrix = graph_sparse_matrix[valid_indices, :][:, valid_indices]

    if cluster_and_subsets is not None:
        cluster_col, allowed_clusters = (
            cluster_and_subsets[0],
            cluster_and_subsets[1],
        )
        valid_indices_bools = np.isin(adata.obs[cluster_col], allowed_clusters)
        valid_indices = np.argwhere(valid_indices_bools).flatten()
        graph_sparse_matrix = graph_sparse_matrix[valid_indices, :][:, valid_indices]

    community_result = cluster_community_from_graph(
        method=method, graph_sparse_matrix=graph_sparse_matrix, directed=directed, **kwargs
    )

    labels = np.zeros(len(adata), dtype=int) + no_community_label

    # No subset required case, use all indices
    if valid_indices is None:
        valid_indices = np.arange(0, len(adata))

    if hasattr(community_result, "membership"):
        labels[valid_indices] = community_result.membership
    else:
        for i, community in enumerate(community_result.communities):
            labels[valid_indices[community]] = i

    # clusters need to be categorical strings
    adata.obs[result_key] = pd.Categorical(labels.astype(str))

    adata.uns[result_key] = {
        "method": method,
        "adj_matrix_key": adj_matrix_key,
        "use_weight": use_weight,
        "layer": layer,
        "layer_conn_type": conn_type,
        "cell_subsets": cell_subsets,
        "cluster_and_subsets": cluster_and_subsets,
        "directed": directed,
    }
    if copy:
        return adata

def louvain_(
    adata: AnnData,
    resolution: float = 1.0,
    use_weight: bool = False,
    weights: Optional[Union[str, Iterable]] = None,
    initial_membership: Optional[List[int]] = None,
    adj_matrix: Optional[sp.csr_matrix] = None,
    adj_matrix_key: Optional[str] = None,
    seed: Optional[int] = None,
    result_key: Optional[str] = None,
    layer: Optional[str] = None,
    obsm_key: Optional[str] = None,
    selected_cluster_subset: Optional[Tuple[str, List[int]]] = None,
    selected_cell_subset: Optional[List[int]] = None,
    directed: bool = True,
    copy: bool = False,
    **kwargs
) -> AnnData:

    kwargs.update(
        {
            "resolution_parameter": resolution,
            "weights": weights,
            "initial_membership": initial_membership,
            "seed": seed,
        }
    )

    return cluster_community(
        adata,
        method="louvain",
        use_weight=use_weight,
        adj_matrix=adj_matrix,
        adj_matrix_key=adj_matrix_key,
        result_key=result_key,
        layer=layer,
        obsm_key=obsm_key,
        cluster_and_subsets=selected_cluster_subset,
        cell_subsets=selected_cell_subset,
        directed=directed,
        copy=copy,
        **kwargs
    )

def scc(
    adata: AnnData,
    spatial_key: str = "spatial",
    key_added: Optional[str] = "scc",
    pca_key: str = "pca",
    e_neigh: int = 30,
    s_neigh: int = 6,
    resolution: Optional[float] = None,
    copy: bool = False,
) -> Optional[AnnData]:

    # Calculate the adjacent matrix
    n_pca_components = 30
    neighbors(adata, X_data=adata.obsm[pca_key], n_neighbors=e_neigh, n_pca_components=n_pca_components)

    # Compute a neighborhood graph of physical space.
    neighbors(
        adata,
        X_data=adata.obsm[spatial_key],
        n_neighbors=s_neigh,
        result_prefix="spatial",
        n_pca_components=n_pca_components,
    )
    
    conn = adata.obsp["connectivities"].copy()
    conn.data[conn.data > 0] = 1
    adj = conn + adata.obsp["spatial_connectivities"]
    adj.data[adj.data > 0] = 1

    # Perform clustering.
    louvain_(adata, adj_matrix=adj, resolution=resolution, result_key=key_added)

    return adata if copy else None
