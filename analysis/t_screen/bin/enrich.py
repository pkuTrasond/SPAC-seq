import gseapy as gp
import pandas as pd
import numpy as np
from goatools.obo_parser import GODag


deg_df = pd.read_csv("~/stereoseq/20240502-SPACseq/DEG.csv", sep="\t")
print(deg_df.dtypes)

clusters = np.unique(deg_df["Cluster"])
degs = [[] for i in range(len(clusters))]
for i, cluster in enumerate(clusters):
    degs[i] = deg_df.loc[deg_df["Cluster"] == cluster]["Gene"]

go_dag = GODag("/home/wpy/stereoseq/20240502-SPACseq/bin/go-basic.obo")

enrs = {}
min_level = 4
min_level_in = 1
for i in range(len(degs)):
    try:
        enr = gp.enrichr(
            gene_list=degs[i],
            #gene_sets=["GO_Biological_Process_2023"],
            #gene_sets=["KEGG_2019_Mouse"],
            gene_sets=["Reactome_2022"],
            background=None,
            organism='Mouse',
            outdir=None,
            no_plot=True
        )
        results = enr.results
        enrs[i] = results
        # enrs[i] = results[results['Term'].apply(lambda term: term.split('(')[1].rstrip(')')).apply(lambda term: go_dag[term].level <= min_level \
        #                                         or (go_dag[term].has_parent("GO:0002376") and go_dag[term].level <= (min_level_in)) \
        #                                         or (go_dag[term].has_parent("GO:0032501") and go_dag[term].level <= (min_level_in)) \
        #                                         or (go_dag[term].has_parent("GO:0008152") and go_dag[term].level <= (min_level_in)) if term in go_dag else False)]

        print(f"Enrichment analysis for list {i} completed successfully:", len(enrs[i]))
        enrs[i].to_csv(f"~/stereoseq/20240502-SPACseq/GO/Cluster_{i}.result", sep="\t", index=False)
    except Exception as e:
        print(f"Error in enrichment analysis for list {i}: {e}")