import pygame
import numpy
import random
import os

from dinosaur_agent import DinosaurAgent
from environment import Environment, Ground, Cloud, Obstacle, Cactus, Ptera

class Game():

    def __init__(self):
        pygame.init()

        self.width = 800
        self.height = 350
        self.fps = 60

        pygame.display.set_caption("NeuroEvolution Training")
        self.window = pygame.display.set_mode((self.width, self.height))
        self.window_color = (247,247,247)
        self.clock = pygame.time.Clock()

        self.cloud_cnt = 60
        self.cloud_threshold = numpy.random.randint(60,100)
        self.obstacle_cnt = 0
        self.obstacle_threshold = numpy.random.randint(40,60)
        self.speed_cnt = 0

        self.obstacles  = []
        self.grounds    = []
        self.clouds     = []

        self.velocity = 15

        self.game_score = 0
        self.add_obstacle()

    def add_clouds(self):
        if self.cloud_cnt == self.cloud_threshold:
            self.clouds.append(Cloud())
            self.cloud_threshold = numpy.random.randint(70,120)
            self.cloud_cnt = 0

    def add_ground(self):
        if len(self.grounds) == 0:
            self.grounds.append(Ground(0, self.velocity))
            self.grounds.append(Ground(self.width, self.velocity))
        if len(self.grounds) == 1:
            self.grounds.append(Ground(self.width, self.velocity))

    def add_obstacle(self):
        if self.obstacle_cnt == self.obstacle_threshold:
            r = random.randrange(0,4)
            if r < 2:
                self.obstacles.append(Cactus(vel = self.velocity))
            else:
                self.obstacles.append(Ptera (vel = self.velocity))

            self.obstacle_cnt = 0
            self.obstacle_threshold = numpy.random.randint(40,60)

    def update_obstacles(self):
        for obstacle in self.obstacles:
            obstacle.update()
            if obstacle.x < (obstacle.width * -1):
                self.obstacles.pop(self.obstacles.index(obstacle))
                self.reward = 1

    def update_clouds(self):
        for cloud in self.clouds:
            cloud.update()
            if cloud.x < (cloud.width * -1):
                self.clouds.pop(self.clouds.index(cloud))

    def update_ground(self):
        for ground in self.grounds:
            ground.update()
            if ground.x < (ground.width * -1):
                self.grounds.pop(self.grounds.index(ground))

    def increment_object_counters(self):
        self.speed_cnt += 1
        self.obstacle_cnt += 1
        self.cloud_cnt += 1

        if self.speed_cnt % 4 == 0:
            self.game_score += 1

    def increment_velocity(self):
        if self.speed_cnt % 1000 == 0:
            self.velocity += 2
            self.speed_cnt = 0

    def close(self):
        pygame.display.quit()
        pygame.quit()
        print('Current score:',self.game_score)
        exit()