import time
import numpy as np
from PIL import Image
from multiprocess import Pool, cpu_count

from vangogh import selection, variation
from vangogh.fitness import drawing_fitness_function, draw_voronoi_image, compute_grid_fitness
from vangogh.population import Population
from vangogh.util import NUM_VARIABLES_PER_POINT, IMAGE_SHRINK_SCALE, REFERENCE_IMAGE

STABILIZATION_PHASE_DURATION = 30


class Evolution:
    def __init__(self,
                 num_points,
                 reference_image: Image,
                 evolution_type='p+o',
                 population_size=200,
                 generation_budget=-1,
                 evaluation_budget=-1,
                 crossover_method="ONE_POINT",
                 mutation_probability='inv_mutable_genotype_length',
                 num_features_mutation_strength=0.25,
                 num_features_mutation_strength_decay=None,
                 num_features_mutation_strength_decay_generations=None,
                 selection_name='tournament_2',
                 initialization='RANDOM',
                 noisy_evaluations=False,
                 verbose=False,
                 generation_reporter=None,
                 enable_spatial_adaptation=False,
                 stabilization_phase_duration=STABILIZATION_PHASE_DURATION,
                 mutation_factor=2.0,
                 seed=0):

        self.reference_image: Image = reference_image.copy()
        self.reference_image.thumbnail(
            (int(self.reference_image.width / IMAGE_SHRINK_SCALE),
             int(self.reference_image.height / IMAGE_SHRINK_SCALE)),
            Image.ANTIALIAS if hasattr(Image, 'ANTIALIAS') else Image.Resampling.LANCZOS
        )
        self.reference_image_array = np.asarray(self.reference_image)

        num_variables = num_points * NUM_VARIABLES_PER_POINT
        feature_intervals = []
        for i in range(num_variables):
            if i % NUM_VARIABLES_PER_POINT == 0:   # X
                feature_intervals.append([0, self.reference_image.width])
            elif i % NUM_VARIABLES_PER_POINT == 1:  # Y
                feature_intervals.append([0, self.reference_image.height])
            else:                                    # R, G, B
                feature_intervals.append([0, 256])

        self.num_points = num_points
        self.feature_intervals = np.array(feature_intervals, dtype=object)
        self.evolution_type = evolution_type
        self.population_size = population_size
        self.generation_budget = generation_budget
        self.evaluation_budget = evaluation_budget
        self.mutation_probability = mutation_probability
        self.num_features_mutation_strength = num_features_mutation_strength
        self.num_features_mutation_strength_decay = num_features_mutation_strength_decay
        self.num_features_mutation_strength_decay_generations = num_features_mutation_strength_decay_generations
        self.selection_name = selection_name
        self.noisy_evaluations = noisy_evaluations
        self.verbose = verbose
        self.generation_reporter = generation_reporter
        self.crossover_method = crossover_method
        self.num_evaluations = 0
        self.initialization = initialization
        self.enable_spatial_adaptation = enable_spatial_adaptation
        self.stabilization_phase_duration = stabilization_phase_duration
        self.mutation_factor = mutation_factor
        self.current_generation = 0
        self.grid_fitness = None

        np.random.seed(seed)
        self.seed = seed

        if 'tournament' in selection_name:
            self.tournament_size = int(selection_name.split('_')[-1])
            if self.population_size % self.tournament_size != 0:
                raise ValueError('The population size must be a multiple of the tournament size')

        self.genotype_length = len(feature_intervals)
        self.population = Population(self.population_size, self.genotype_length, self.initialization)
        self.elite = None
        self.elite_fitness = np.inf

        if mutation_probability == 'inv_genotype_length':
            self.mutation_probability = 1 / self.genotype_length
        elif mutation_probability == 'inv_mutable_genotype_length':
            self.mutation_probability = 1 / self.genotype_length

        if self.evolution_type == 'p+o' and self.noisy_evaluations:
            raise ValueError("Using P+O is not compatible with noisy evaluations")

    def __update_elite(self, population):
        best_fitness_idx = np.argmin(population.fitnesses)
        best_fitness = population.fitnesses[best_fitness_idx]
        if self.noisy_evaluations or best_fitness < self.elite_fitness:
            self.elite = population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

    def __compute_grid_critique(self):
        if self.enable_spatial_adaptation and len(self.population.genes) > 0:
            best_idx = np.argmin(self.population.fitnesses)
            grid_cell_size = max(1, self.reference_image.width // 11)
            self.grid_fitness, _ = compute_grid_fitness(
                self.population.genes[best_idx], self.reference_image,
                grid_cell_size=grid_cell_size
            )

    def __classic_generation(self, merge_parent_offspring=False):
        offspring = Population(self.population_size, self.genotype_length, self.initialization)
        offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()

        offspring.genes = variation.crossover(offspring.genes, self.crossover_method)

        use_spatial = (
            self.enable_spatial_adaptation
            and self.current_generation >= self.stabilization_phase_duration
            and self.grid_fitness is not None
        )

        offspring.genes = variation.mutate(
            offspring.genes,
            self.feature_intervals,
            mutation_probability=self.mutation_probability,
            num_features_mutation_strength=self.num_features_mutation_strength,
            grid_fitness=self.grid_fitness if use_spatial else None,
            grid_cell_size=max(1, self.reference_image.width // 11) if use_spatial else None,
            mutation_factor=self.mutation_factor
        )

        offspring.fitnesses = drawing_fitness_function(offspring.genes, self.reference_image)
        self.num_evaluations += len(offspring.genes)

        self.__update_elite(offspring)

        if merge_parent_offspring:
            self.population.stack(offspring)
        else:
            self.population = offspring

        self.population = selection.select(
            self.population, self.population_size, selection_name=self.selection_name
        )

        self.__compute_grid_critique()

    def run(self):
        data = []

        self.population.initialize(self.feature_intervals)
        self.population.fitnesses = drawing_fitness_function(
            self.population.genes, self.reference_image
        )
        self.num_evaluations = len(self.population.genes)

        self.__update_elite(self.population)
        self.__compute_grid_critique()

        start_time_seconds = time.time()
        i_gen = 0

        while True:
            if self.num_features_mutation_strength_decay_generations is not None:
                if i_gen in self.num_features_mutation_strength_decay_generations:
                    self.num_features_mutation_strength *= self.num_features_mutation_strength_decay

            if self.evolution_type == 'classic':
                self.__classic_generation(merge_parent_offspring=False)
            elif self.evolution_type == 'p+o':
                self.__classic_generation(merge_parent_offspring=True)
            else:
                raise ValueError('unknown evolution type:', self.evolution_type)

            i_gen += 1
            self.current_generation = i_gen

            if self.verbose:
                print('generation:', i_gen,
                      'best fitness:', self.elite_fitness,
                      'avg. fitness:', np.mean(self.population.fitnesses))

            data.append({
                "num-generations": i_gen,
                "num-evaluations": self.num_evaluations,
                "time-elapsed": time.time() - start_time_seconds,
                "best-fitness": self.elite_fitness,
                "crossover-method": self.crossover_method,
                "population-size": self.population_size,
                "num-points": self.num_points,
                "initialization": self.initialization,
                "seed": self.seed
            })

            if self.generation_reporter is not None:
                self.generation_reporter(
                    {"num-generations": i_gen,
                     "num-evaluations": self.num_evaluations,
                     "time-elapsed": time.time() - start_time_seconds},
                    self
                )

            if 0 < self.generation_budget <= i_gen:
                break
            if 0 < self.evaluation_budget <= self.num_evaluations:
                break
            if self.population.is_converged():
                break

        draw_voronoi_image(
            self.elite,
            self.reference_image.width,
            self.reference_image.height,
            scale=IMAGE_SHRINK_SCALE
        ).save(
            f"./img/van_gogh_final_{self.seed}_{self.population_size}_{self.crossover_method}"
            f"_{self.num_points}_{self.initialization}_{self.generation_budget}.png"
        )

        return data


if __name__ == '__main__':
    evo = Evolution(100,
                    REFERENCE_IMAGE,
                    evolution_type='p+o',
                    population_size=100,
                    generation_budget=300,
                    crossover_method='ONE_POINT',
                    initialization='RANDOM',
                    num_features_mutation_strength=0.25,
                    selection_name='tournament_4',
                    noisy_evaluations=False,
                    verbose=True)
    evo.run()