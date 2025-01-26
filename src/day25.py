"""One line that may or may not intersect.
AKA The Chaos Game
Genome file from: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/
Gene file from: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/
"""

import matplotlib.pylab as plt
from skimage.draw import line
from pyranges import read_gtf
import pandas as pd
import numpy as np
import py2bit


def prepare_gene_data(gtf_path="rsc/hg38.knownGene.gtf.gz"):
    gene_data = read_gtf(gtf_path).as_df()
    gene_data = (
        gene_data.query("Feature == 'transcript'")
        .query("Source == 'knownGene'")
        .query("Chromosome in @main_chroms")
    )
    gene_data["length"] = gene_data["End"] - gene_data["Start"]
    gene_data = gene_data[~gene_data["gene_name"].isna()]
    gene_data.to_csv("rsc/hg38.filtered_genes.csv")


def load_gene_data(csv_path: str = "rsc/hg38.filtered_genes.csv"):
    return pd.read_csv(csv_path, index_col=0)


SIZE = 1000
main_chroms = [f"chr{v}" for v in list(range(1, 23)) + ["X", "Y"]]
all_gene_data = load_gene_data()


def create_pattern(rng_obj):
    # Get random gene sequence
    with py2bit.open("rsc/hg38.2bit") as genome_file:
        filtered_genes = all_gene_data.query("length > 1e5").query("length < 1e6")
        gene_info = filtered_genes.iloc[rng_obj.integers(0, len(filtered_genes)), :]
        chrom, start, end, gene_name, gene_length = gene_info[
            ["Chromosome", "Start", "End", "gene_id", "length"]
        ]
        gene_sequence = genome_file.sequence(str(chrom), int(start), int(end))
    # Chaos game time!
    print(gene_length, gene_name)
    base_positions = {
        "A": np.array([0, 0]),
        "C": np.array([0, 1]),
        "G": np.array([1, 1]),
        "T": np.array([1, 0]),
    }
    pos = np.array([0.5, 0.5])
    show_grid = np.ones(shape=(SIZE, SIZE))
    for base in gene_sequence:
        new_pos = (pos + base_positions[base]) / 2
        yy, xx = line(*(pos * SIZE).astype(int), *(new_pos * SIZE).astype(int))
        yy, xx = np.minimum(yy, SIZE - 1), np.minimum(xx, SIZE - 1)
        show_grid[yy, xx] *= 0.99
        pos = new_pos
    return show_grid


rng = np.random.default_rng(1701)
fig, axs = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
for ax in axs.flatten():
    seed = rng.integers(0, 1e5)
    ax.imshow(create_pattern(np.random.default_rng(seed)), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("out/day25.png", bbox_inches="tight", transparent=True)
plt.show()
