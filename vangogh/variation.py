import numpy as np


def crossover(genes, method="ONE_POINT"):
    parents_1 = np.vstack((genes[:len(genes) // 2], genes[:len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2:], genes[len(genes) // 2:]))

    if method == "ONE_POINT":
        crossover_points = np.random.randint(0, genes.shape[1], size=genes.shape[0])
        offspring = np.zeros(shape=genes.shape, dtype=int)

        for i in range(len(genes)):
            offspring[i,:] = np.where(np.arange(genes.shape[1]) <= crossover_points[i], parents_1[i,:], parents_2[i,:])
    elif method == "UNIVARIATE":
        offspring = univariate_crossover(genes, "UNIVARIATE")
    elif method == "SPLIT_POS_COLOR":
        offspring = split_position_color_crossover(genes)
    elif method == "SPLIT_POS_COLOR_ONE_POINT":
        offspring = split_position_color_crossover_one_point_color(genes)
    else:
        raise Exception("Unknown crossover method")

    return offspring


def univariate_crossover(genes, method="UNIVARIATE"):
    parents_1 = np.vstack((genes[:len(genes) // 2], genes[:len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2:], genes[len(genes) // 2:]))

    if method == "UNIVARIATE":
        crossover_mask = np.random.choice([True, False], size=genes.shape)
        offspring = np.zeros(shape=genes.shape, dtype=int)

        offspring = np.where(crossover_mask, parents_1, parents_2)
    else:
        raise Exception("Unknown crossover method")

    return offspring


def split_position_color_crossover(genes):
    n_individuals, n_features = genes.shape

    if n_features % 5 != 0:
        raise ValueError(
            f"Expected genotype length divisible by 5, got length {n_features}."
        )

    n_points = n_features // 5

    points = genes.reshape(n_individuals, n_points, 5)

    position_genes = points[:, :, 0:2].reshape(n_individuals, n_points * 2)
    color_genes = points[:, :, 2:5].reshape(n_individuals, n_points * 3)

    crossed_positions = crossover(
        position_genes,
        "ONE_POINT"
    )

    crossed_colors = univariate_crossover(
        color_genes,
        "UNIVARIATE"
    )

    offspring_points = np.empty_like(points)

    offspring_points[:, :, 0:2] = crossed_positions.reshape(
        n_individuals,
        n_points,
        2
    )

    offspring_points[:, :, 2:5] = crossed_colors.reshape(
        n_individuals,
        n_points,
        3
    )

    return offspring_points.reshape(n_individuals, n_features).astype(
        genes.dtype,
        copy=False
    )


def split_position_color_crossover_one_point_color(genes):
    n_individuals, n_features = genes.shape

    if n_features % 5 != 0:
        raise ValueError(
            f"Expected genotype length divisible by 5, got length {n_features}."
        )

    n_points = n_features // 5

    points = genes.reshape(n_individuals, n_points, 5)

    position_genes = points[:, :, 0:2].reshape(n_individuals, n_points * 2)
    color_genes = points[:, :, 2:5].reshape(n_individuals, n_points * 3)

    crossed_positions = crossover(
        position_genes,
        "ONE_POINT"
    )

    crossed_colors = crossover(
        color_genes,
        "ONE_POINT"
    )

    offspring_points = np.empty_like(points)

    offspring_points[:, :, 0:2] = crossed_positions.reshape(
        n_individuals,
        n_points,
        2
    )

    offspring_points[:, :, 2:5] = crossed_colors.reshape(
        n_individuals,
        n_points,
        3
    )

    return offspring_points.reshape(n_individuals, n_features).astype(
        genes.dtype,
        copy=False
    )


def mutate(genes, feature_intervals,
           mutation_probability=0.1, num_features_mutation_strength=0.05):
    mask_mut = np.random.choice([True, False], size=genes.shape,
                                p=[mutation_probability, 1 - mutation_probability])

    mutations = generate_plausible_mutations(genes, feature_intervals,
                                             num_features_mutation_strength)

    offspring = np.where(mask_mut, mutations, genes)

    return offspring


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

        # Fix out-of-range
        mutations[:, i] = np.where(mutations[:, i] > feature_intervals[i][1],
                                   feature_intervals[i][1], mutations[:, i])
        mutations[:, i] = np.where(mutations[:, i] < feature_intervals[i][0],
                                   feature_intervals[i][0], mutations[:, i])

    mutations = mutations.astype(int)
    return mutations