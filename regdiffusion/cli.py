import argparse
import os
import numpy as np
import scanpy as sc
from .trainer import RegDiffusionTrainer

def main():
    parser = argparse.ArgumentParser(
        description="Infer a gene regulatory network (GRN) from a single-cell count dataset."
    )
    parser.add_argument(
        "input",
        help="Input single-cell count dataset file (CSV or H5AD format)."
    )
    parser.add_argument(
        "--output",
        default="rd_grn.csv",
        help="Output file path for the edgelist (CSV). Default: rd_grn.csv"
    )
    parser.add_argument(
        "--top_gene_percentile",
        type=int,
        default=50,
        help="Percentile cutoff to filter weak edges (e.g., 50 for the top 50%%). Default: 50"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=-1,
        help="Number of edges per gene to extract (-1 for all edges). Default: -1"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers to use for edgelist extraction. Default: 4"
    )

    args = parser.parse_args()

    if args.input.endswith('.csv'):
        adata = sc.read_csv(args.input)
    elif args.input.endswith('.h5ad'):
        adata = sc.read_h5ad(args.input)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or H5AD file.")

    print("Applying log transformation on the dataset...")
    x = adata.X
    if (x < 0).sum() > 0:
        raise ValueError("Negative values spotted in the data. Please only provide count data in this CLI tool. ")
    if (x - x.astype(int)).sum() > 1:
        raise ValueError("Data might have been log transformed. Please only provide count data in this CLI tool. ")
    x = np.log(x + 1.0)

    print("Initializing RegDiffusion trainer and starting training...")
    rd_trainer = RegDiffusionTrainer(x)
    rd_trainer.train()

    grn = rd_trainer.get_grn(
        adata.var_names, 
        top_gene_percentile=args.top_gene_percentile)

    print("Extracting edgelist from GRN...")
    edgelist = grn.extract_edgelist(k=args.k, workers=args.workers)

    edgelist.to_csv(args.output, index=False)
    print(f"Edgelist successfully saved to {args.output}")

if __name__ == "__main__":
    main()