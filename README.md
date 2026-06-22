# Evolutionary Van Gogh

To run this code, install the required packages from requirements.txt and open the Jupyter notebook (`analysis.ipynb`) for an example of how to run the code and experiments with it. The code was tested with Python version 3.9.5, more recent versions may need changes to the requirements.

```bash
pip install -r requirements.txt
```

To start a Jupyter notebook instance, have Jupyter notebook [installed](https://jupyter.org/install#jupyter-notebook) and start it up in this directory.

---

To access the repo use `git clone https://github.com/lxmampuya/Evolutionary_VanGogh.git`.

## Hypothesis 1

- Hierarchical EA where we start with fewer points in earlier generation and increase it later. See hier_evo.py for code changes.
1. Switch to the branch `git switch hypo1-hierarchical`
2. Open the notebook analysis_h1.ipynb. You may load the saved data (df object) with `df = pd.read_pickle("0604_evolution_results.pkl")`. This is also in a cell near the bottom of the notebook.
3. Re-run the cells if desired to see training process, it takes 1-2 minutes to run each.
4. The generated images results are saved in img/h1/. The numbers of each folder correspond to the experiments numbered in the notebook.

---

## Hypothesis 2

- Mutation Decay for both the mutation strength and the mutation probability
1. Switch to the branch `git switch decay-mutation-rate`
2. The full experiment is in `analysis.ipynb` including the figures used in the presentation
3. If you wish to also run the random search you can change `random_search = False` into `random_search = True` in the third cell and run the notebook again

---

## Hypothesis 3

Hypothesis 3
- Uniform crossover applied to the spatial and color genes

To run the code, do the following:

Switch to the branch `git switch split-position-color-ga`
The full experiment is in analysis.ipynb including the figures used in the presentation
Set TEST_CASE to either 1, 2 or 3, where:
1: Runs Baseline vs Split (Uniform)
2: Runs Baseline vs Split (Uniform) vs Split (One Point)
3. Runs Baseline vs Split (Uniform) vs No Split (Uniform)


Our hypothesis was that applying univariate crossover to the color genes would improve convergence because color information can be optimized independently for each cell. Spatial genes are somewhat dependent on one another: the effect of a cell's position depends on the positions of neighboring cells, and changing a single coordinate can significantly alter the boundaries and shapes of multiple regions, although in the case of one-point crossover on the unordered list, it is unclear how much of the points would stay together during variation.

In contrast, color genes are independent from each other. A Voronoi cell can often benefit from inheriting a better red, green, or blue value from one parent without requiring corresponding changes to neighbouring cells. Small adjustments to individual color channels can improve the match to the target image while leaving the underlying spatial structure unchanged. Because of this relative independence, we expected gene-wise recombination to be more effective for colors, allowing offspring to combine useful color components from both parents at a much finer level than one-point crossover.

We therefore split the genome into spatial and color components, applying one-point crossover to XY and uniform crossover to RGB separately. The experimental results supported this hypothesis, as the split spatial/color method using univariate crossover over the colors achieved improved fitness of 26.19% compared to the baseline. 

To verify that this improvement was mainly caused by the univariate color crossover, we also tested a variant where the color genes used one-point crossover instead. This produced little improvement over the baseline. To further investigate the effect, we also applied uniform crossover to the original, unsplit individuals — producing results statistically indistinguishable from the split model. This may be explained by the random ordering of points in the genome: since spatial neighbors are not necessarily adjacent in the gene sequence, there are few positional dependencies for one-point crossover to exploit.

We propose that uniform crossover's per-gene independence allows each point to optimize its position and color freely, whereas one-point crossover tends to inherit  adjacent genes together, constraining individual optimization.

## Hypothesis 4
 
- Spatial adaptation: Voronoi points in high-error image regions receive stronger mutation pressure than points in regions that already look good.

1. Switch to the branch `git switch adaptive_spatial_ga`
2. Open the notebook `analysis_h4.ipynb` for an example of how to run the experiments.

The idea is that not all regions of the image are equally hard to reconstruct. After each generation the image is divided into a grid and the average pixel error per cell is computed. During mutation, each Voronoi point looks up which grid cell it sits in and gets its mutation probability and strength scaled proportionally to the error there. Points in bad regions explore more aggressively, points in already-good regions are left mostly alone.
 
There is a stabilization phase at the start where the algorithm runs plain mutation first, giving the population time to form a rough approximation of local colors before spatial targeting kicks in to adjust more for geometric errors. The best fitness achieved was below 37000 using 100 Voronoi points.

