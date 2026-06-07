# Evolutionary Van Gogh

To run this code, install the required packages from requirements.txt and open the Jupyter notebook (`analysis.ipynb`) to see our experimental results. The code was tested with Python version 3.9.5, more recent versions may need changes to the requirements.

```bash
pip install -r requirements.txt
```

To start a Jupyter notebook instance, have Jupyter notebook [installed](https://jupyter.org/install#jupyter-notebook) and start it up in this directory.

## Hypothesis 3
### Uniform crossover on color genes

Our hypothesis was that applying univariate crossover to the color genes would improve convergence because color information can be optimized independently for each cell. Spatial genes are somewhat dependent on one another: the effect of a cell's position depends on the positions of neighboring cells, and changing a single coordinate can significantly alter the boundaries and shapes of multiple regions, although in the case of one-point crossover on the unordered list, it is unclear how much of the points would stay together during variation.

In contrast, color genes are independent from each other. A Voronoi cell can often benefit from inheriting a better red, green, or blue value from one parent without requiring corresponding changes to neighbouring cells. Small adjustments to individual color channels can improve the match to the target image while leaving the underlying spatial structure unchanged. Because of this relative independence, we expected gene-wise recombination to be more effective for colors, allowing offspring to combine useful color components from both parents at a much finer level than one-point crossover.

We therefore split the genome into spatial and color components, applying one-point crossover to XY and uniform crossover to RGB separately. The experimental results supported this hypothesis, as the split spatial/color method using univariate crossover over the colors achieved improved fitness of 26.19% compared to the baseline. 

To verify that this improvement was mainly caused by the univariate color crossover, we also tested a variant where the color genes used one-point crossover instead. This produced little improvement over the baseline. To further investigate the effect, we also applied uniform crossover to the original, unsplit individuals — producing results statistically indistinguishable from the split model. This may be explained by the random ordering of points in the genome: since spatial neighbors are not necessarily adjacent in the gene sequence, there are few positional dependencies for one-point crossover to exploit.

We propose that uniform crossover's per-gene independence allows each point to optimize its position and color freely, whereas one-point crossover tends to inherit  adjacent genes together, constraining individual optimization.
