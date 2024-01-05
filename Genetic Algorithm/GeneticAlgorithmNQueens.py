from random import uniform
from numpy.random import randint

TOURNAMENT_SIZE = 5
CHROMOSOME_LENGTH = 8

class Individual:

    def __init__(self):
        # list data structure containing indexes of queens. Look at the __repr__ method for more info
        self.chromosome = [randint(0, CHROMOSOME_LENGTH) for i in range(CHROMOSOME_LENGTH)]

    def fitness(self):
        """"
        Assume c is number of collisions between queens
        fitness = 1 / (c + 1)
        """
        collisions = 0

        # Check rows
        for i in range(len(self.chromosome)):
            for j in range(i + 1, len(self.chromosome)):
                if self.chromosome[i] == self.chromosome[j]:
                    collisions += 1

        # Check diagonals
        for i in range(len(self.chromosome)):
            for j in range(i + 1, len(self.chromosome)):
                if abs(i - j) == abs(self.chromosome[i] - self.chromosome[j]):
                    collisions += 1

        return 1 / (collisions + 1)

    def set_gene(self, index, value):
        self.chromosome[index] = value

    def __repr__(self):
        """
        Represents the individual's chromosome as a chessboard with queens.

        Each gene in the chromosome represents the column position of the queen in its respective row.
        The chessboard is displayed as an 8x8 grid, where 'Q' represents a queen and '.' represents an empty square.

        Returns:
        str: A string representation of the chessboard.
        """
        board = []
        for row in range(CHROMOSOME_LENGTH):
            row_str = ['.' for _ in range(CHROMOSOME_LENGTH)]
            # Place the queen in the position indicated by the chromosome
            row_str[self.chromosome[row]] = 'Q'
            # Join the row into a string and add it to the board
            board.append(' '.join(row_str))

        return '\n'.join(board)

class Population:

    def __init__(self, size):
        self.size = size
        self.individuals = [Individual() for i in range(size)]

    # Linear search for finding individual with highest fitness
    def get_fittest(self):
        fittest = self.individuals[0]

        for i in range(self.size):
            if fittest.fitness() <= self.individuals[i].fitness():
                fittest = self.individuals[i]

        return fittest

    # Return n fittest individuals
    def get_fittest_elitism(self, n):
        self.individuals.sort(key=lambda x: x.fitness(), reverse=True)
        return self.individuals[:n]

    def get_size(self):
        return self.size

    def get_individual(self, index):
        return self.individuals[index]

    def save_individual(self, index, individual):
        self.individuals[index] = individual

class GeneticAlgorithm:
    def __init__(self, population_size = 100, crossover_rate=0.65, mutation_rate=0.15, elitism_param=5):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_param = elitism_param

    def run(self):
        pop = Population(self.population_size)
        generation_count = 0

        while pop.get_fittest().fitness() != 1: # fitness is 1 when no collisions
            generation_count += 1
            print("Generation: %d Fittest: %f" % (generation_count, pop.get_fittest().fitness()))

            pop = self.evolve_population(pop)

        print("Solution found in generation %d" % generation_count)
        print(pop.get_fittest())

    def evolve_population(self, pop):
        next_population = Population(self.population_size)

        # Due to elitism first n fittest individuals are added to next population
        next_population.individuals.extend(pop.get_fittest_elitism(self.elitism_param))

        # crossover
        for index in range(self.elitism_param, next_population.get_size()):
            first = self.random_selection(pop)
            second = self.random_selection(pop)
            next_population.save_individual(index, self.crossover(first, second))

        # mutation
        for individual in next_population.individuals:
            self.mutate(individual)

        return next_population

    def crossover(self, offspring1, offspring2):
        """
        Performs crossover between two offspring to create a new individual.

        This method combines parts of the chromosomes from two parent individuals to create a new
        individual. It randomly selects a start and end point in the chromosome for crossover.
        The new individual's chromosome is then composed of genes from both parents: genes before
        the start point are taken from the first parent, genes between start and end are taken
        from the second parent, and genes after the end point are from the first parent again.

        Parameters:
        offspring1 (Individual): The first parent individual.
        offspring2 (Individual): The second parent individual.

        Returns:
        Individual: The new individual created as a result of crossover.

        Note:
        CHROMOSOME_LENGTH is the length of the chromosome.
        """
        cross_individual = Individual()

        start = randint(CHROMOSOME_LENGTH)
        end = randint(CHROMOSOME_LENGTH)

        if start > end:
            start, end = end, start

        cross_individual.chromosome = offspring1.chromosome[:start] + offspring2.chromosome[start:end] + offspring1.chromosome[end:]

        return cross_individual

    def mutate(self, individual):
        """
        Mutates an individual's chromosome based on a predefined mutation rate.

        This method iterates over each gene in the individual's chromosome. For each gene, it decides
        whether to mutate based on a randomly generated number and the predefined mutation rate. If
        the condition for mutation is met, the gene is altered to a new value.

        Parameters:
        individual (Individual): The individual whose chromosome is to be mutated.

        Note:
        CHROMOSOME_LENGTH defines the length of the chromosome, i.e., the number of genes in it.
        self.mutation_rate is a float between 0 and 1, representing the probability of any gene mutating.
        """
        for index in range(CHROMOSOME_LENGTH):
            if uniform(0, 1) <= self.mutation_rate:
                individual.chromosome[index] = randint(CHROMOSOME_LENGTH)

    # this is called tournament selection
    def random_selection(self, actual_population):
        """
        Performs random selection from the given population to create a new population for the next generation.

        The method implements a tournament selection strategy, where a subset of individuals is chosen
        randomly from the actual population, and the fittest individual from this subset is selected
        to be a part of the new generation. This process is repeated until the new population is complete.

        Parameters:
        actual_population (Population): The current population from which individuals are to be selected.

        Returns:
        Individual: The fittest individual from the newly created population.
        """
        new_population = Population(TOURNAMENT_SIZE)

        for i in range(new_population.get_size()):
            random_index = randint(new_population.get_size())
            new_population.save_individual(i, actual_population.get_individual(random_index))

        return new_population.get_fittest()


if __name__ == '__main__':
    algorithm = GeneticAlgorithm(100)
    algorithm.run()