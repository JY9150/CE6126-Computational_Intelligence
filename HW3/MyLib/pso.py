import numpy as np
from typing import Callable

class Particle():
    def __init__(self, dimension) -> None:
        self.position = np.random.rand(dimension)
        self.velocity = np.random.rand(dimension)
        self.local_best_position = self.position
        self.local_best_fitness = float('-inf')



class PSO():
    def __init__(self, particle_dimension: int, num_particles: int, num_iteration: int, fitness_func: Callable, ro1: float = 0.5, ro2: float = 0.5) -> None:
        self.dimension = particle_dimension
        self.num_particles = num_particles
        self.num_iteration = num_iteration
        self.fitness_func = fitness_func
        self.ro1 = ro1
        self.ro2 = ro2

        self.particles = [Particle(particle_dimension) for _ in range(num_particles)]
        self.global_best_position = np.random.rand(particle_dimension)
        self.global_best_fitness = float('-inf')
    
    def run(self) -> np.ndarray:
        for _ in range(self.num_iteration):
            for particle in self.particles:
                particle_fitness = self.fitness_func(particle.position)

                if particle_fitness > self.global_best_fitness:
                    self.global_best_position = particle.position
                    self.global_best_fitness = particle_fitness
                if particle_fitness > particle.local_best_fitness:
                    particle.local_best_position = particle.position
                    particle.local_best_fitness = particle_fitness
                    
                particle.velocity = particle.velocity + self.ro1 * (particle.local_best_position - particle.position) + self.ro2 * (self.global_best_position - particle.position)
                
                # add velocity limit if needed

                particle.position = particle.position + particle.velocity
            
        return self.global_best_position