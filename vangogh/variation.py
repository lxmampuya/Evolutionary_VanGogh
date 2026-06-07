import numpy as np

NUM_VARIABLES_PER_POINT = 5


def crossover(genes, method="ONE_POINT"):
    parents_1 = np.vstack((genes[:len(genes) // 2], genes[:len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2:], genes[len(genes) // 2:]))

    if method == "ONE_POINT":
        crossover_points = np.random.randint(0, genes.shape[1], size=genes.shape[0])
        offspring = np.zeros(shape=genes.shape, dtype=int)
        for i in range(len(genes)):
            offspring[i, :] = np.where(
                np.arange(genes.shape[1]) <= crossover_points[i],
                parents_1[i, :],
                parents_2[i, :]
            )
    elif method == "UNIFORM":
        num_points = genes.shape[1] // NUM_VARIABLES_PER_POINT
        point_mask = np.random.choice([True, False], size=(genes.shape[0], num_points))
        gene_mask = np.repeat(point_mask, NUM_VARIABLES_PER_POINT, axis=1)
        offspring = np.where(gene_mask, parents_1, parents_2)
    else:
        raise Exception("Unknown crossover method")

    return offspring


def mutate(genes, feature_intervals,
           mutation_probability=0.1,
           num_features_mutation_strength=0.05,
           grid_fitness=None,
           grid_cell_size=None,
           mutation_factor=2.0):
    """
    Vectorized mutation with optional spatial weighting.
    When grid_fitness is provided, points in high-error regions
    receive stronger mutation probability and magnitude.
    """
    if grid_fitness is None or grid_cell_size is None:
        # Original fast vectorized path — no spatial weighting
        mask_mut = np.random.choice(
            [True, False], size=genes.shape,
            p=[mutation_probability, 1 - mutation_probability]
        )
        mutations = generate_plausible_mutations(
            genes, feature_intervals, num_features_mutation_strength
        )
        return np.where(mask_mut, mutations, genes)

    # Spatial path — per-point mutation strength based on grid error
    new_genes = genes.copy()
    num_individuals, genotype_length = genes.shape
    num_points = genotype_length // NUM_VARIABLES_PER_POINT
    max_grid_error = max(float(grid_fitness.max()), 1e-8)

    for ind_idx in range(num_individuals):
        for p_idx in range(num_points):
            base_idx = p_idx * NUM_VARIABLES_PER_POINT
            x = new_genes[ind_idx, base_idx]
            y = new_genes[ind_idx, base_idx + 1]

            cx = int(x // grid_cell_size)
            cy = int(y // grid_cell_size)

            p_mut = mutation_probability
            s_mut = num_features_mutation_strength

            if 0 <= cy < grid_fitness.shape[0] and 0 <= cx < grid_fitness.shape[1]:
                normalized_error = grid_fitness[cy, cx] / max_grid_error
                multiplier = 1.0 + mutation_factor * normalized_error
                p_mut = min(0.95, mutation_probability * multiplier)
                s_mut = num_features_mutation_strength * multiplier

            for v_offset in range(NUM_VARIABLES_PER_POINT):
                if np.random.rand() < p_mut:
                    interval = feature_intervals[base_idx + v_offset]
                    span = interval[1] - interval[0]
                    mutated_val = int(
                        new_genes[ind_idx, base_idx + v_offset] +
                        np.random.normal(0, s_mut * span)
                    )
                    new_genes[ind_idx, base_idx + v_offset] = np.clip(
                        mutated_val, interval[0], interval[1]
                    )

    return new_genes


def generate_plausible_mutations(genes, feature_intervals,
                                 num_features_mutation_strength=0.25):
    mutations = np.zeros(shape=genes.shape)
    for i in range(genes.shape[1]):
        range_num = feature_intervals[i][1] - feature_intervals[i][0]
        low = -num_features_mutation_strength / 2
        high = +num_features_mutation_strength / 2
        mutations[:, i] = range_num * np.random.uniform(low=low, high=high,
                                                         size=mutations.shape[0])
        mutations[:, i] += genes[:, i]
        mutations[:, i] = np.where(mutations[:, i] > feature_intervals[i][1],
                                   feature_intervals[i][1], mutations[:, i])
        mutations[:, i] = np.where(mutations[:, i] < feature_intervals[i][0],
                                   feature_intervals[i][0], mutations[:, i])
    return mutations.astype(int)