import numpy
from dinosaur_agent import DinosaurAgent

class GeneticAlgorithm():

    @staticmethod
    def initialize(population_size):

        agents_list = [DinosaurAgent() for x in range(population_size)]
        active_agents = [agent for agent in agents_list]

        return agents_list, active_agents

    @staticmethod
    def perform_selection(agents):

        # Calculate fitness for each agent

        for agent in agents:
            agent.score = agent.score ** 2

        sum = 0
        for agent in agents:
            sum += agent.score
        for agent in agents:
            agent.fitness = agent.score / sum

        # Create a crossing pool
        pool = []
        for agent in agents:
            f = int(agent.fitness * len(agents) * 10)
            for i in range(f):
                pool.append(agent)

        pool = sorted(pool, key=lambda a: a.fitness)[::-1]               # Sort agents in pool by fitness (descending)
        pool = pool[0:(len(pool) // 10)]                                 # Perform natural selection (10%)

        return pool

    @staticmethod
    def crossover(x, y):

        cross_gene = {}
        heritage_probability = 0.1 * numpy.random.randint(11)

        for index in x.keys():
            cross_gene[index] = numpy.copy(x[index])

            original_shape = x[index].shape
            for i in range(original_shape[0]):
                for j in range(original_shape[1]):
                    if numpy.random.random() < heritage_probability:
                        x[index][i, j] = y[index][i, j]

        return cross_gene

    @staticmethod
    def mutate(gene):

        def mutate_genome(genome):
            original_shape = genome.shape
            for i in range(original_shape[0]):
                for j in range(original_shape[1]):
                    if numpy.random.random() < mutation_rate:
                        genome[i, j] = numpy.random.randn() * mutation_magnitude

            return genome.reshape(original_shape)

        mutation_rate = 0.05                # Lower mutation rate - faster initail, slower long term
        mutation_magnitude = 0.1

        mutated_gene = {}

        for i in gene.keys(): # Mutate the DNA
            mutated_gene[i] = mutate_genome(numpy.copy(gene[i]))

        return mutated_gene

    @staticmethod
    def reproduce(population_size, pool):
        agents_list = []
        active_agents = []

        for i in range(population_size):

            # Randomly select 2 genes for crossover
            a = numpy.random.randint(0, len(pool))
            b = numpy.random.randint(0, len(pool))

            x = {}
            y = {}

            for i in pool[0].neural_network.neurons.keys():
                x[i] = numpy.copy(pool[a].neural_network.neurons[i])
                y[i] = numpy.copy(pool[b].neural_network.neurons[i])

            # Crossover and Mutation
            cross_gene = GeneticAlgorithm.crossover(x, y)
            child_gene = GeneticAlgorithm.mutate(cross_gene)

            # New child
            child = DinosaurAgent()
            for i in child.neural_network.neurons.keys():
                child.neural_network.neurons[i] = child_gene[i]

            agents_list.append(child)
            active_agents.append(child)

        return agents_list, active_agents