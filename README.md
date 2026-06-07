# The spatial adaptation hypothesis

## Explanation

The main thing we added on top of the basic algorithm is a spatial adaptation mechanism. The hypothesis is:

> Not all regions of the image are equally hard to reconstruct. Wasting mutation budget on regions that already look good is inefficient. Instead, Voronoi points that sit in high-error regions of the image should be mutated more aggressively, while points in already-good regions should be left mostly alone.

To do this, after each generation we divide the image into a grid and compute the average pixel error per cell. This gives us a map of where the image looks bad. Then during mutation, we look up which grid cell each Voronoi point falls into and scale its mutation probability and mutation strength proportionally to the error in that cell.

In practice this means a point sitting in a blurry or wrong-colored region will explore more — it moves around more and tries more colors. A point in a region that already looks right barely gets touched. The idea is to concentrate the search budget where it actually matters.

There is a stabilization phase at the start (first 50 generations by default) where we run plain mutation without spatial adaptation. This gives the population time to settle into a rough approximation of the image before we start targeting specific regions. If we apply spatial adaptation too early the population hasn't converged enough to have meaningful error patterns yet.

---

## Key settings

| Setting | What it does |
|---|---|
| `num_points` | Number of Voronoi points per individual. More points = finer detail but slower |
| `population_size` | How many individuals evolve in parallel |
| `mutation_probability` | Base chance of mutating each gene |
| `num_features_mutation_strength` | How big each mutation step is, as a fraction of the value range |
| `selection_name` | Tournament size — `tournament_2` is gentler, `tournament_4` is more aggressive |
| `enable_spatial_adaptation` | Turns the spatial allocator on or off |
| `stabilization_phase_duration` | How many generations to run before spatial adaptation kicks in |
| `mutation_factor` | Controls the mutation multiplier per grid cell. The formula is `1 + mutation_factor × normalized_error`, so every cell above zero error gets some boost — a cell at 15% error gets 1.6x stronger mutation, a cell at max error gets 5x. There are basically no neutral cells in practice. |

---

## What we found

The plain algorithm already works reasonably well but tends to converge early — the population all starts looking similar before it has found a great solution. The spatial adaptation helps with this because it keeps pushing on the hard regions even after the easy ones are solved, which gives the algorithm more useful work to do in later generations.

The tricky balance is between `mutation_factor` and tournament size. If mutation is too aggressive and selection pressure is too high at the same time, diversity collapses fast and you end up stuck in a local optimum. The sweet spot we found was `tournament_2` with `mutation_factor` around 3–4, which keeps enough diversity alive for the spatial adaptation to actually help.

We also tested uniform crossover against one-point crossover. Uniform crossover performed better in our runs — instead of splitting parents at one point and taking a left/right chunk from each, uniform crossover randomly picks each Voronoi point independently from either parent. This means offspring get a proper mix of good points from across the whole image rather than inheriting one half from each parent, which matters a lot when different regions of the image have been solved by different individuals.

The best fitness we achieved was below 37000, which is a pretty solid reconstruction considering we're using 100 Voronoi points to approximate the entire image.

---

## A note on how this was built

The code and debugging for this project was done with help from Claude (Anthropic's AI assistant). Claude helped identify bugs in the codebase, suggested the vectorized Voronoi rendering, and helped think through the spatial adaptation logic and parameter tuning. The hypothesis and experimental direction were our own — Claude was more like a pair programmer than an author.

---