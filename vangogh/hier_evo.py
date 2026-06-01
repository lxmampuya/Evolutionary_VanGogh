"""Hypothesis1: Hierarchical Evolution
Start with a small number of points, then increase the number of points after some generations, defined by a schedule. This allows the algorithm to first find a good global structure with fewer points, then refine the details with more points. Coarse-to-fine optimization can help avoid getting stucked early on when the search space is large.
"""

from operator import ne
import time
import numpy as np

from vangogh import selection, variation
from vangogh.fitness import drawing_fitness_function, draw_voronoi_image
from vangogh.population import Population
from vangogh.util import NUM_VARIABLES_PER_POINT, IMAGE_SHRINK_SCALE, REFERENCE_IMAGE
from .evolution import Evolution

class H_evolution(Evolution):
    def __init__(self, *args, schedule=None, **kwargs):
        """_summary_

        Parameters
        ----------
        schedule : List[Tuple[float, int]]
            First tuple is when to change num_point, second tuple is the new num_point. e.g. [(0,50), (0.25,100)] means start with num_point=50, then at 25% of the total generations, change num_point to 100.
        """
        super().__init__(*args, **kwargs)
        self.schedule = schedule
        assert self.schedule[0][1] == self.num_points, "First num_point in schedule must match initial num_point"

    def __duplicate_points(self, merge_parent_offspring=False, new_num_points=None):
        assert new_num_points, "new_num_points must be provided"
        #! update genotype_length and feature_intervals based on new_num_points
        num_variables = new_num_points * NUM_VARIABLES_PER_POINT
        feature_intervals = []
        for i in range(num_variables):
            if i % NUM_VARIABLES_PER_POINT == 0:  # X
                feature_intervals.append([0, self.reference_image.width])
            elif i % NUM_VARIABLES_PER_POINT == 1:  # Y
                feature_intervals.append([0, self.reference_image.height])
            else:  # color (RGBA)
                feature_intervals.append([0, 256])
        
        self.num_points = new_num_points
        self.feature_intervals = np.array(feature_intervals, dtype=object)
        old_len = self.genotype_length # 250 for 50points
        self.genotype_length = len(feature_intervals)
        new_len = self.genotype_length # 500 for 100 points

        # 1. create offspring population
        offspring = Population(self.population_size, self.genotype_length, self.initialization)
        #! Duplicate points to update length of each offspring
        # 2. Figure out how many full clean copies we can fit
        num_duplicates = new_len // old_len
        remainder = new_len % old_len

        # 3. Create the full clean duplicates. pop.genes is (population_size by genotype_length)
        if num_duplicates > 0:
            duplicated_genes = np.tile(self.population.genes, (1, num_duplicates))
        else:
            duplicated_genes = np.empty((self.population_size, 0), dtype=int)

        # 4. Handle the remainder: sample indices without replacement
        if remainder > 0:
            # Pick random gene indices from the original genotype length without replacement
            random_indices = np.random.choice(old_len, size=remainder, replace=False)
            # Extract those columns
            remainder_genes = self.population.genes[:, random_indices]
            # Concatenate the clean duplicates and the random remainder
            offspring.genes[:] = np.hstack((duplicated_genes, remainder_genes))
        else:
            offspring.genes[:] = duplicated_genes
        
        # offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()
        # variation
        offspring.genes = variation.crossover(offspring.genes, self.crossover_method)
        offspring.genes = variation.mutate(offspring.genes, self.feature_intervals,
                                           mutation_probability=self.mutation_probability,
                                           num_features_mutation_strength=self.num_features_mutation_strength)
        # evaluate offspring
        offspring.fitnesses = drawing_fitness_function(offspring.genes,
                                                       self.reference_image)
        self.num_evaluations += len(offspring.genes)

        self._update_elite(offspring)

        # selection
        if merge_parent_offspring:
            # p+o mode
            self.population.stack(offspring)
        else:
            # just replace the entire thing
            self.population = offspring

        self.population = selection.select(self.population, self.population_size,
                                           selection_name=self.selection_name)

    def run(self):
        # num_points affects feature_intervals,genotype_length
        print("---Running H-evolution---")
        print(self.schedule)
        # Mostly from evolution.run(), with some modifications to support hierarchical evolution
        data = []
        self.population.initialize(self.feature_intervals)

        self.population.fitnesses = drawing_fitness_function(
            self.population.genes, self.reference_image)
        self.num_evaluations = len(self.population.genes)

        best_fitness_idx = np.argmin(self.population.fitnesses)
        best_fitness = self.population.fitnesses[best_fitness_idx]
        if best_fitness > self.elite_fitness:
            self.elite = self.population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

        start_time_seconds = time.time()

        # run generation_budget
        i_gen = 0
        #! assume we at least have 2 entries in schedule, first one is for initial num_points, second one is for the first update
        self.schedule.pop(0)
        gen_to_update, new_num_points = self.schedule.pop(0)
        gen_to_update = int(gen_to_update * self.generation_budget)
        while True:
            if self.num_features_mutation_strength_decay_generations is not None:
                if i_gen in self.num_features_mutation_strength_decay_generations:
                    self.num_features_mutation_strength *= self.num_features_mutation_strength_decay
            
            #! check schedule
            if i_gen == gen_to_update:
                print(f"--Generations: {i_gen}, updating num_points to {new_num_points}--")
                if self.evolution_type == 'classic':
                    self.__duplicate_points(merge_parent_offspring=False, new_num_points=new_num_points)
                elif self.evolution_type == 'p+o':
                    self.__duplicate_points(merge_parent_offspring=True, new_num_points=new_num_points)
                else:
                    raise ValueError('unknown evolution type:', self.evolution_type)
                if len(self.schedule) > 0:
                    gen_to_update, new_num_points = self.schedule.pop(0)
            #! normal updates
            else:
                if self.evolution_type == 'classic':
                    self._classic_generation(merge_parent_offspring=False)
                elif self.evolution_type == 'p+o':
                    self._classic_generation(merge_parent_offspring=True)
                else:
                    raise ValueError('unknown evolution type:', self.evolution_type)

            # generation terminated
            i_gen += 1
            if self.verbose:
                if i_gen % 10 == 0:
                    print('generation:', i_gen, 'best fitness:', self.elite_fitness, 'avg. fitness:',
                      np.mean(self.population.fitnesses))

            data.append({"num-generations": i_gen,
                         "num-evaluations": self.num_evaluations,
                         "time-elapsed": time.time() - start_time_seconds,
                         "best-fitness": self.elite_fitness,
                         "crossover-method": self.crossover_method,
                         "population-size": self.population_size, "num-points": self.num_points,
                         "initialization": self.initialization,
                         "seed": self.seed})
            if self.generation_reporter is not None:
                self.generation_reporter(
                    {"num-generations": i_gen, "num-evaluations": self.num_evaluations,
                     "time-elapsed": time.time() - start_time_seconds}, self)

            if 0 < self.generation_budget <= i_gen:
                break
            if 0 < self.evaluation_budget <= self.num_evaluations:
                break

            # check if evolution should terminate because optimum reached or population converged
            if self.population.is_converged():
                break

        draw_voronoi_image(self.elite, self.reference_image.width, self.reference_image.height,
                           scale=IMAGE_SHRINK_SCALE) \
            .save(
            f"./img/van_gogh_final_{self.seed}_{self.population_size}_{self.crossover_method}_{self.num_points}_{self.initialization}_{self.generation_budget}.png")
        return data