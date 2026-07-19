# Visualizing Inferred GRN

RegDiffusion hands you back an adjacency matrix. On a 5,000-gene run that is
25 million numbers, and no amount of squinting at a heatmap will tell you
whether the network learned anything biological. The fastest way to build (or
lose) confidence in an inferred network is to *look* at it and ask whether
genes that belong together ended up together.

This tutorial walks through that end to end on a real dataset: pull healthy
human PBMCs from [CELLxGENE](https://cellxgene.cziscience.com/), infer a GRN
over 5,000 genes, annotate the genes with
[Reactome](https://reactome.org/) pathways, and render the result with
[`lightgraph`](https://haozhu233.github.io/lightgraph/).

```{note}
Network plots in `regdiffusion` are drawn by `lightgraph`, which renders on an
HTML canvas with viewport culling. This matters more than it sounds: the older
`pyvis`/`vis.js` backend stalls for the better part of a minute before the
first frame at 5,000 nodes, which effectively ruled out looking at a whole
network at once. `lightgraph` stays interactive at that size, so the
whole-network view below is something you can actually pan around in.
```

## Step 1. Get the data

We use the NYGC multimodal PBMC dataset from
[Hao et al., *Cell* 2021](https://doi.org/10.1016/j.cell.2021.04.048) — the
Seurat v4 CITE-seq reference, 161,764 healthy human PBMCs, deeply annotated,
and a de facto reference for immune cell types. It is available through the
[CELLxGENE Census](https://cellxgene.cziscience.com/collections/b0cf0afa-ec40-4d65-b570-ed4ceacc6813)
API:

```python
>>> import cellxgene_census
>>> import scanpy as sc
>>> import numpy as np
>>>
>>> DATASET_ID = "ed5d841d-6346-47d4-ab2f-7119ad7e3a35"
>>>
>>> with cellxgene_census.open_soma(census_version="2023-12-15") as census:
>>>     adata = cellxgene_census.get_anndata(
>>>         census,
>>>         organism="Homo sapiens",
>>>         obs_value_filter=f"dataset_id == '{DATASET_ID}' and is_primary_data == True",
>>>         obs_column_names=["cell_type", "assay", "donor_id", "disease"],
>>>     )
>>> adata.shape
(161764, 60664)
```

Census ships raw counts indexed by Ensembl ID, so we switch to gene symbols
(which is what pathway databases speak) and trim the gene families that soak up
variance without carrying much regulatory signal:

```python
>>> adata.var["ensembl_id"] = adata.var_names
>>> adata.var_names = adata.var["feature_name"].astype(str)
>>> adata.var_names_make_unique()
>>>
>>> mask = ~(
>>>     adata.var_names.str.startswith("MT-")
>>>     | adata.var_names.str.startswith(("RPL", "RPS"))
>>>     | adata.var_names.str.match(r"^HB[ABDEGQZ]\d?$")
>>> )
>>> adata = adata[:, mask].copy()
>>> sc.pp.filter_genes(adata, min_cells=int(0.01 * adata.n_obs))
>>>
>>> sc.pp.normalize_total(adata, target_sum=1e4)
>>> sc.pp.log1p(adata)
>>> sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat")
>>> adata = adata[:, adata.var.highly_variable].copy()
>>> adata.shape
(161764, 5000)
```

## Step 2. Infer the network

Nothing special here — this is the standard RegDiffusion workflow. Passing
`cell_types` lets the model condition on the 30 annotated cell types in this
dataset.

```python
>>> import regdiffusion as rd
>>>
>>> X = np.asarray(adata.X.todense(), dtype=np.float32)
>>> codes = adata.obs["cell_type"].astype("category").cat.codes.values.astype(np.int64)
>>>
>>> trainer = rd.RegDiffusionTrainer(X, cell_types=codes, n_steps=1000, seed=123)
>>> trainer.train()
>>> grn = trainer.get_grn(adata.var_names.to_numpy())
>>> grn.to_hdf5("pbmc_nygc_grn.hdf5", as_sparse=True)
```

On an RTX PRO 6000 the 1,000 training steps take about 13 seconds; loading and
densifying the expression matrix takes considerably longer than the training
itself.

A first sanity check before any plotting — who are the hub genes?

```python
>>> from lightgraph import degree
>>>
>>> edges = grn.extract_edgelist(k=2)
>>> edges = edges.assign(weight=edges.weight.abs())
>>> deg = degree(edges[["source", "target", "weight"]].values.tolist())
>>> sorted(deg.items(), key=lambda kv: -kv[1])[:12]
[('LTB', 306.0), ('IL7R', 292.0), ('ETS1', 203.0), ('CD3D', 181.0),
 ('NKG7', 149.0), ('CD3E', 145.0), ('LDHB', 144.0), ('CD79A', 138.0),
 ('EEF1B2', 138.0), ('MS4A1', 131.0), ('CST7', 129.0), ('EEF1A1', 115.0)]
```

That is a reassuring list: T cell identity (`CD3D`, `CD3E`, `IL7R`, `LTB`), B
cell identity (`CD79A`, `MS4A1`), cytotoxic effectors (`NKG7`, `CST7`), and the
lymphoid transcription factor `ETS1`. The network put PBMC lineage markers at
the center without being told anything about lineages.

## Step 3. Group genes by function

Topology is much easier to judge when nodes carry an independent functional
label. We use Reactome's pathway hierarchy: every gene is annotated to one or
more pathways, and each pathway rolls up to one of ~30 top-level pathways like
*Immune System* or *Cell Cycle*. Rolling annotations up to that top level gives
a small, readable palette.

Two files from Reactome are all we need — the gene sets and the parent/child
relations:

```bash
wget https://reactome.org/download/current/ReactomePathways.gmt.zip
wget https://reactome.org/download/current/ReactomePathwaysRelation.txt
unzip ReactomePathways.gmt.zip
```

```python
>>> from collections import defaultdict, Counter
>>>
>>> name, gene2path = {}, defaultdict(set)
>>> for line in open("ReactomePathways.gmt"):
>>>     f = line.rstrip("\n").split("\t")
>>>     name[f[1]] = f[0]
>>>     for g in f[2:]:
>>>         gene2path[g].add(f[1])
>>>
>>> parent, children = defaultdict(set), defaultdict(set)
>>> for line in open("ReactomePathwaysRelation.txt"):
>>>     p, c = line.rstrip("\n").split("\t")
>>>     if p.startswith("R-HSA") and c.startswith("R-HSA"):
>>>         parent[c].add(p)
>>>         children[p].add(c)
>>>
>>> roots = {p for p in set(parent) | set(children) | set(name) if not parent[p]}
>>>
>>> def ancestors(pid, cache={}):
>>>     if pid not in cache:
>>>         cache[pid] = ({pid} if pid in roots or not parent[pid]
>>>                       else set().union(*(ancestors(p) for p in parent[pid])))
>>>     return cache[pid]
```

A gene often reaches several top-level pathways. We let the roots vote — the
one covering most of the gene's pathways wins — and break ties toward the
*smaller* root, so genes do not all avalanche into `Metabolism` and
`Signal Transduction`:

```python
>>> root_size = Counter()
>>> for g, pids in gene2path.items():
>>>     for r in set().union(*(ancestors(p) for p in pids)):
>>>         root_size[r] += 1
>>>
>>> def assign(symbols):
>>>     out = {}
>>>     for s in symbols:
>>>         votes = Counter()
>>>         for p in gene2path.get(s, ()):
>>>             for a in ancestors(p):
>>>                 votes[a] += 1
>>>         if votes:
>>>             best = min(votes, key=lambda r: (-votes[r], root_size[r], r))
>>>             out[s] = name.get(best, best)
>>>     return out
>>>
>>> groups_all = assign(grn.gene_names)
>>> len(groups_all), len(grn.gene_names)
(2678, 5000)
```

Roughly **54% of our genes get a Reactome annotation**. That is normal and worth
being upfront about: highly variable genes in single cell data are full of
lncRNAs, immunoglobulin and TCR segments, and poorly characterized ORFs that no
curated database covers. Keep the twelve largest groups and simply leave
everything else out of the dictionary — genes missing from `node_group_dict`
render in a neutral gray and stay out of the legend, which is much more honest
than dumping half the network into a colored bucket called "Other":

```python
>>> top12 = [g for g, _ in Counter(groups_all.values()).most_common(12)]
>>> groups = {g: p for g, p in groups_all.items() if p in top12}
>>> Counter(groups.values()).most_common()
[('Immune System', 403), ('Signal Transduction', 366),
 ('Gene expression (Transcription)', 338), ('Metabolism', 335),
 ('Metabolism of proteins', 179), ('Disease', 154), ('Cell Cycle', 142),
 ('Metabolism of RNA', 125), ('Vesicle-mediated transport', 76),
 ('Transport of small molecules', 70), ('DNA Repair', 66),
 ('Developmental Biology', 64)]
```

### Give the pathways fixed, meaningful colors

This tutorial shows several views of the same network — the whole graph, a
local neighborhood, and the same analysis rerun on a different cell type. By
default `lightgraph` assigns palette colors in sorted-group order, so a group
that is absent from one view shifts the colors of everything after it and the
figures stop being comparable.

Pin the colors instead. Assigning hues by *biological family* also means
related pathways read as related — the two genome-maintenance groups are both
violet, the three metabolism groups are all green, and so on:

```python
>>> PATHWAY_COLORS = {
>>>     # defense and pathology
>>>     'Immune System':                   '#e34948',
>>>     'Disease':                         '#ff756f',
>>>     # stress / alert
>>>     'Cellular responses to stimuli':   '#eda100',
>>>     # genome maintenance
>>>     'Cell Cycle':                      '#4a3aa7',
>>>     'DNA Repair':                      '#918bff',
>>>     # metabolism
>>>     'Metabolism':                      '#008300',
>>>     'Metabolism of proteins':          '#36a231',
>>>     'Metabolism of RNA':               '#56bf50',
>>>     # information flow
>>>     'Gene expression (Transcription)': '#2a78d6',
>>>     # signaling
>>>     'Signal Transduction':             '#e87ba4',
>>>     # trafficking
>>>     'Vesicle-mediated transport':      '#eb6834',
>>>     'Transport of small molecules':    '#ff8552',
>>>     # development
>>>     'Developmental Biology':           '#1baf7a',
>>> }
>>> GROUP_ORDER = list(PATHWAY_COLORS)   # union across every figure below
```

Pass both to every figure. `group_order` reserves a palette slot for each
group even when that group is missing from the current subset, so nothing
shifts:

```python
>>> COMMON = dict(group_colors=PATHWAY_COLORS, group_order=GROUP_ORDER)
```

```{note}
Thirteen categories is more than color alone can safely carry. Under
red-green color vision deficiency some of these pairs are genuinely
indistinguishable, and no palette fixes that — it is a limit of the channel,
not of the color choices. The design leans on that deliberately: confusions
are pushed *inside* a family, where they cost least (mistaking `Metabolism of
RNA` for `Metabolism of proteins` is far less damaging than mistaking either
for `Cell Cycle`). Identity is never carried by color alone in these figures —
every group is named with its size in the legend, clicking a legend entry
isolates that group, hovering a node names its group, and the search box finds
genes directly.
```

## Step 4. Look at the whole network

Now we can put all 5,000 genes on screen at once. We keep the top 2 edges per
gene (10,000 edges), color by Reactome group, and size each node by its
centrality so the regulatory hubs are visible at a glance.

PageRank is the natural centrality here — cheap to compute even at this size,
and it agrees with the degree ranking on the top hubs. One catch: PageRank on
this graph spans roughly 375× from the smallest gene to the largest, with a
very long tail. `net_vis` min-max normalizes the metric linearly, so feeding
the raw values would leave the median gene at 0.002 of the size range — i.e.
everything minimum-size except a handful of hubs. Taking the log first spreads
the middle of the distribution across the size channel:

```python
>>> import numpy as np
>>> from lightgraph import net_vis, pagerank
>>>
>>> pr = pagerank(edges[["source", "target", "weight"]].values.tolist())
>>> centrality = {g: float(np.log10(v)) for g, v in pr.items()}
>>>
>>> v = net_vis(
>>>     edges=edges,
>>>     node_groups=groups,
>>>     node_metric=centrality,
>>>     metric_map="size",
>>>     metric_label="PageRank (log)",
>>>     metric_size_range=(5, 32),
>>>     node_color="#e0e0e0",            # unannotated genes
>>>     edge_weight_to_opacity=True,
>>>     weight_opacity_range=(0.008, 0.10),
>>>     simulation_strength=12000,
>>>     link_distance=45,
>>>     warmup_ticks=600,
>>>     show_labels=False,
>>>     show_ellipses=False,
>>>     show_statistics=True,
>>>     height="720px",
>>>     **COMMON,                        # pinned pathway colors
>>> )
>>> v.save("grn_pbmc_full.html")
```

With 10,000 edges behind 5,000 sizeable nodes, the edge layer has to get out of
the way — hence the very low `weight_opacity_range`. The edges still do their
job of holding the layout together and darkening where the graph is dense,
without turning the picture into a gray wash.

The top genes by PageRank are the same lineage markers the degree ranking
found: `IL7R`, `LTB`, `ETS1`, `CD3D`, `NKG7`, `CD3E`, `CST7`, `CD79A`, `MS4A1`.

```{warning}
Note what we are *not* doing: `lightgraph` has a `group_attraction` parameter
that pulls same-group nodes toward a shared centroid. Turning it up would make
the pathway groups look beautifully separated — but it would be manufacturing
that separation from the labels rather than revealing it from the edges. We
leave it at its default so that any clustering you see comes from the inferred
network alone.
```

<iframe src="_static/grn_pbmc_full.html" width="100%" height="740" style="border:1px solid #ddd;"></iframe>

Things worth trying in the widget above: hover a node to fade everything
outside its immediate neighborhood, double-click a node to isolate its
neighborhood entirely (double-click empty space or hit `Escape` to come back),
and click a legend entry to focus one pathway group. Labels are off by default
at this size — zoom in and they appear.

### Does the picture actually mean anything?

Eyeballing a hairball is a good way to fool yourself, so it is worth checking
the impression numerically. If functionally related genes really are wired
together, edges should connect same-group genes more often than chance. Shuffle
the pathway labels among annotated genes and compare:

```python
>>> both = edges[edges.source.isin(groups) & edges.target.isin(groups)]
>>> sg, tg = both.source.map(groups), both.target.map(groups)
>>> observed = (sg == tg).mean()
>>>
>>> rng = np.random.default_rng(0)
>>> keys, vals = list(groups), list(groups.values())
>>> null = np.mean([
>>>     (both.source.map(m) == both.target.map(m)).mean()
>>>     for m in (dict(zip(keys, rng.permutation(vals))) for _ in range(200))
>>> ])
>>> observed, null, observed / null
(0.197, 0.118, 1.67)
```

Same-group edges are **1.67× more common than chance** (z ≈ 10). Broken down by
group — observed vs. its own shuffled null — the effect is very unevenly
distributed:

| Reactome group | within-group | null | ratio |
| --- | --- | --- | --- |
| DNA Repair | 0.052 | 0.013 | **3.95** |
| Cell Cycle | 0.094 | 0.030 | **3.10** |
| Immune System | 0.217 | 0.094 | **2.32** |
| Vesicle-mediated transport | 0.024 | 0.014 | 1.63 |
| Disease | 0.053 | 0.034 | 1.59 |
| Signal Transduction | 0.127 | 0.085 | 1.49 |
| Transport of small molecules | 0.019 | 0.014 | 1.38 |
| Metabolism of RNA | 0.023 | 0.026 | 0.89 |
| Metabolism of proteins | 0.035 | 0.040 | 0.87 |
| Metabolism | 0.066 | 0.077 | 0.86 |
| Gene expression (Transcription) | 0.064 | 0.079 | 0.80 |
| Developmental Biology | 0.007 | 0.012 | 0.55 |

The coherent groups are exactly the ones that *should* be coherent in resting
PBMCs: DNA repair and cell cycle are tightly co-regulated programs, and immune
function is the biology this tissue is built around. The groups at or below 1.0
are the broad grab-bag categories — `Metabolism` and
`Gene expression (Transcription)` each cover hundreds of genes spanning
unrelated processes, so there is no reason for them to occupy one region of the
graph, and they do not.

In other words, the network is not uniformly "functionally organized". It is
organized around specific, tightly co-regulated programs, and the visualization
is showing you something real rather than a layout artifact.

### Should you filter out weak edges?

Worth being clear about what `extract_edgelist(k=2)` does: it is a *per-gene
rank* filter, not a strength filter. Every gene contributes its 2 best edges
whether those edges are strong or barely there. Since the weight distribution is
heavily skewed, that lets a lot of weak links in:

```python
>>> np.quantile(edges.weight, [0, 0.25, 0.5, 0.75, 0.9, 1])
array([ 1.136,  1.545,  2.102,  3.988,  7.578, 15.32 ])
```

Adding a weight threshold on top sharpens the functional signal considerably —
here is the same-group enrichment from above, recomputed at increasing cutoffs:

| cutoff | edges | genes kept | enrichment |
| --- | --- | --- | --- |
| none | 10,000 | 5,000 | 1.67× |
| 25th pct (1.55) | 7,504 | 3,921 | 1.78× |
| median (2.10) | 5,007 | 2,602 | 2.00× |
| 75th pct (3.99) | 2,502 | 1,291 | 2.64× |
| 90th pct (7.58) | 1,001 | 525 | 3.23× |

It is tempting to read that column as "strong edges are simply more
trustworthy" and threshold aggressively. Don't — look at *which* genes survive
before you believe it. Here is how each pathway group is thinned at each cutoff
(percentage of the group's genes still on the plot):

| Reactome group | none | median | 75th pct | 90th pct |
| --- | --- | --- | --- | --- |
| Immune System | 403 | 331 (82%) | 255 (63%) | 145 (36%) |
| Signal Transduction | 366 | 268 (73%) | 164 (45%) | 64 (17%) |
| Disease | 154 | 103 (67%) | 56 (36%) | 27 (18%) |
| Developmental Biology | 64 | 44 (69%) | 28 (44%) | 13 (20%) |
| Metabolism | 335 | 165 (49%) | 74 (22%) | 29 (9%) |
| Gene expression (Transcription) | 338 | 124 (37%) | 49 (14%) | 17 (5%) |
| **Cell Cycle** | 142 | 57 (40%) | 22 (15%) | **2 (1%)** |
| **DNA Repair** | 66 | 26 (39%) | 4 (6%) | **1 (2%)** |

Weight filtering is **not** biologically neutral. It preferentially keeps the
dominant program of the tissue — in resting PBMCs, immune identity genes carry
by far the largest weights — and it deletes Cell Cycle and DNA Repair almost
entirely by the 90th percentile. Those are precisely the two groups with the
*highest* functional coherence in the table above (3.10× and 3.95×).

Which means the rising global enrichment is partly an artifact: the graph is
collapsing onto one large, internally-consistent immune module, and "same-group
edges" rises because one group is swallowing the plot. You are not making the
picture more trustworthy so much as making it more about T and B cells.

### Reduce `k` instead of thresholding

There is a better lever, and it is the one `extract_edgelist` already gives
you. Note what `k` actually does: for each *target* gene it keeps that gene's
`k` strongest regulators. It is a **rank** filter applied per gene, so it is
invariant to the scale of any particular gene's weights — a cell cycle gene
keeps its best regulators even if those weights are small in absolute terms.
(This is the same primitive as `select_top_n_per_target` in
[flashscenic](https://github.com/haozhu233/flashscenic), which builds SCENIC
modules on top of RegDiffusion and unions several rank-based selections
precisely so that no single global cutoff decides what survives.)

Compare the two ways of getting down to about 5,000 edges:

| | edges | genes kept | Immune System | Cell Cycle | DNA Repair | enrichment |
| --- | --- | --- | --- | --- | --- | --- |
| `extract_edgelist(k=1)` | 5,000 | **5,000** | 403 | **142** | **66** | 1.71× |
| `k=2` + median weight cut | 5,007 | 2,602 | 331 | 57 | 26 | 2.05× |

Same edge budget. The rank-based version keeps **every gene and every pathway
group completely intact**; the weight cut throws away half the genes and 60% of
the cell cycle. The weight cut does score a somewhat higher enrichment (2.05×
vs 1.71×) — but as established above, a good part of that is the compositional
artifact of the immune program crowding out everything else, not a cleaner
graph.

So, practical guidance:

- **To make the plot sparser** — lower `k`. It is scale-free, it costs you
  nothing in gene coverage, and no program silently vanishes.
- **To de-emphasize weak links without deleting them** — keep every edge and
  let opacity carry the weight (`edge_weight_to_opacity=True`, as used above).
- **To threshold on weight anyway** — check your program of interest survives
  first, and say so in the caption, because the composition of the figure
  changed and not only its density.

`GRN` has this built in, applied to the adjacency matrix before you extract
anything:

```python
>>> grn.remove_weak_edges(threshold=2.1)
>>> edges_strong = grn.extract_edgelist(k=2)
```

Note that `remove_weak_edges` mutates the `GRN` in place and only ever raises
the cutoff, so reload from HDF5 if you want to go back to a weaker threshold.

## Step 5. Zoom in on a neighborhood

Whole-network views are good for gestalt, bad for specifics. For a claim you
can check gene by gene, use `visualize_local_neighborhood`, which pulls the
2.5-hop neighborhood around genes you name. Let us take the two B cell hubs
from the degree ranking:

```python
>>> g = grn.visualize_local_neighborhood(
>>>     ["MS4A1", "CD79A"], k=20, node_group_dict=groups, height="640px",
>>>     **COMMON,
>>> )
>>> g.save("grn_pbmc_bcell.html")
```

<iframe src="_static/grn_pbmc_bcell.html" width="100%" height="660" style="border:1px solid #ddd;"></iframe>

The two genes you asked about are drawn at double size, and edge width follows
the absolute regulatory strength of each link. The neighborhood comes back with
16 genes:

```
BANK1, CD22, CD79A, CD79B, HLA-DQA1, IGHD, IGHM, IGKC, LINC00926,
MEF2C, MS4A1, PAX5, RALGPS2, TCL1A, TNFRSF13C, VPREB3
```

Every one of these is a B cell gene. `CD79A`/`CD79B`/`CD22`/`MS4A1` are B cell
receptor complex and surface markers; `IGHD`/`IGHM`/`IGKC` are immunoglobulin
chains; `PAX5` and `MEF2C` are the master transcription factors of B cell
commitment; `TCL1A` and `VPREB3` mark naive and pre-B stages; `TNFRSF13C`
(BAFF-R) drives B cell survival. Not a single T cell, myeloid, or NK gene got
pulled in, even though the run had 5,000 genes to choose from and no cell type
labels were used to build the neighborhood.

Note also that `LINC00926` and `RALGPS2` sit right in this module. Neither
carries a Reactome annotation, so both render in the neutral gray — but their
company is a strong hint, and the literature bears it out.
[`LINC00926` is a lncRNA whose expression is restricted to the B cell
lineage](https://doi.org/10.3324/haematol.2025.287524) (highest in naive B
cells, down-regulated on activation), and the Human Protein Atlas classifies
[`RALGPS2` as B cell group enriched](https://www.proteinatlas.org/ENSG00000116191-RALGPS2/immune+cell)
among immune cells. This is the practical payoff of an inferred GRN: it places
genes your annotation database has nothing to say about right next to the ones
it does.

## Step 6. A different cell type, a different network

Everything above was inferred across all 30 PBMC cell types at once, so the
strongest signal in the data is lineage identity — which is why the hubs are T
and B cell markers. A good way to check that the method is reading biology
rather than reproducing one fixed answer is to run it inside a *single* cell
type, where lineage identity is constant and cannot dominate.

CD14+ monocytes are the largest population here (42,690 cells) and about as far
from the T/B axis as this dataset goes. The pipeline is identical, with two
changes: highly variable genes are re-selected *within* monocytes, so they
capture variation across monocyte states rather than between lineages, and
there is no cell type covariate left to condition on.

```python
>>> with cellxgene_census.open_soma(census_version="2023-12-15") as census:
>>>     mono = cellxgene_census.get_anndata(
>>>         census, organism="Homo sapiens",
>>>         obs_value_filter=(f"dataset_id == '{DATASET_ID}' and is_primary_data == True "
>>>                           "and cell_type == 'CD14-positive monocyte'"),
>>>     )
>>> # ...same gene filtering, normalization and HVG selection as Step 1...
>>> trainer = rd.RegDiffusionTrainer(X_mono, n_steps=1000, seed=123)   # no cell_types
>>> trainer.train()
>>> mono_grn = trainer.get_grn(mono.var_names.to_numpy())
```

The monocyte figure below is rendered with the same `**COMMON` colors, which is
what makes it comparable to the whole-PBMC one: `Immune System` is the same red
in both, `Cell Cycle` the same violet. Two groups differ between the runs —
`DNA Repair` drops out of the monocyte top twelve and `Cellular responses to
stimuli` enters — and because `group_order` reserves a slot per group, that
swap changes nothing else on screen.

Only 2,739 of the 5,000 genes overlap with the all-PBMC run, so this is
genuinely a different gene space. The hubs change completely:

| | top genes by PageRank |
| --- | --- |
| All PBMC | `IL7R`, `LTB`, `ETS1`, `CD3D`, `NKG7`, `CD3E`, `CST7`, `CD79A`, `MS4A1` |
| CD14+ monocytes | `S100A8`, `TXNIP`, `S100A9`, `IL1B`, `HLA-DPA1`, `HLA-DPB1`, `MNDA`, `CD74`, `HLA-DRA`, `S100A12`, `VCAN` |

Not a single gene in common, and the monocyte list is textbook myeloid biology:
the `S100A8`/`S100A9`/`S100A12` alarmins, `IL1B` for inflammatory activation,
and the MHC class II machinery (`HLA-DRA`, `HLA-DPA1`, `HLA-DPB1`, `CD74`) that
monocytes use to present antigen.

<iframe src="_static/grn_mono_full.html" width="100%" height="740" style="border:1px solid #ddd;"></iframe>

The pathway coherence analysis from Step 4 tells the sharper story. Overall
enrichment is essentially unchanged — 1.63× in monocytes vs 1.67× across all
PBMCs — so both networks are organized to the *same degree*. But **which**
programs are coherent is almost entirely different:

| Reactome group | monocytes | all PBMC |
| --- | --- | --- |
| Cellular responses to stimuli | **4.64×** | not in top 12 |
| Transport of small molecules | **2.71×** | 1.28× |
| Immune System | 2.16× | 2.27× |
| Metabolism of RNA | 1.46× | 0.92× |
| Signal Transduction | 1.33× | 1.52× |
| **Cell Cycle** | **1.00×** | **3.12×** |
| DNA Repair | too few edges | 4.12× |

Cell cycle coherence collapses from 3.12× to exactly chance, and DNA repair
does not even have enough edges left to test. That is the correct answer:
circulating monocytes are terminally differentiated and essentially
non-proliferating, so there is no coordinated cell cycle program for the model
to find. What replaces it at the top is *Cellular responses to stimuli* at
4.64× — stress and stimulus response, which is precisely what distinguishes one
circulating monocyte from another.

The method is not applying a fixed template. Run it on a mixed population with
proliferating cells and cell cycle comes out coherent; run it inside a
post-mitotic population and that structure correctly disappears.

```{warning}
Two of the monocyte hubs are worth treating with suspicion rather than
excitement. `XIST` is the X-inactivation lncRNA — it is a hub because the
donors differ in sex, not because of monocyte regulation. `MTRNR2L8` is a
mitochondrially-derived transcript that frequently reflects technical
variation. Neither was regressed out here. Whenever a GRN hub is a sex-linked
or mitochondrial gene, check whether it is tracking a covariate of your donors
before building a story on it.
```

## Where to go next

- Swap the Reactome roll-up for a finer level of the hierarchy when you zoom
  into a neighborhood — top-level pathways are coarse, and a 16-gene module
  deserves more resolution.
- Pass `node_group_dict='auto'` to let `lightgraph` detect communities from
  topology alone, then check how well those communities line up with your
  functional annotation.
- Raise `k` in `extract_edgelist` for a denser whole-network view. `lightgraph`
  will keep up well past the 10,000 edges used here.
- The full `lightgraph` API (analytics, layouts, styling, export) is documented
  in its [Python vignette](https://haozhu233.github.io/lightgraph/vignette_python.html).
