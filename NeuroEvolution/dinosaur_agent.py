import numpy
import pygame
import os

from neural_network import NeuralNetwork

class Const():

    DINO_WIDTH              = 59
    DINO_HEIGHT             = 63
    DINO_OFFSET_TOP         = 237
    DINO_OFFSET_LEFT        = 50

    DINO_WIDTH_DUCKING      = 79
    DINO_HEIGHT_DUCKING     = 40
    DINO_OFFSET_TOP_DUCKING = 260

    DISTANCE_OFFSET_X       = 870
    DISTANCE_OFFSET_Y       = 400

class DinosaurAgent():

    def __init__(self):

        # Resources path
        self.resources_path = os.getcwd() + '/sprites/'

        # Load running images
        self.frame_run = 0
        self.running_images = []
        self.running_images.append(pygame.image.load(self.resources_path + 'dino_run_0.png'))
        self.running_images.append(pygame.image.load(self.resources_path + 'dino_run_1.png'))

        # Load ducking images
        self.frame_duck = 0
        self.ducking_images = []
        self.ducking_images.append(pygame.image.load(self.resources_path + 'dino_duck_0.png'))
        self.ducking_images.append(pygame.image.load(self.resources_path + 'dino_duck_1.png'))

        # Set color
        self.color = (196, 6, 6)


        # Initialize properties
        self.x = Const.DINO_OFFSET_LEFT
        self.y = Const.DINO_OFFSET_TOP
        self.w = Const.DINO_WIDTH
        self.h = Const.DINO_HEIGHT

        self.score = 0
        self.fitness = 0

        # Initialize states
        self.is_running = True
        self.is_jumping = False
        self.is_ducking = False

        # Initialize neural network
        self.neural_network = NeuralNetwork()
        self.jump_count_const = 8
        self.jump_count_running_var = self.jump_count_const


    def draw(self, window):

        # RUNNING
        if self.y == Const.DINO_OFFSET_TOP:
            if self.frame_run >= 16:
                self.frame_run = 0
            window.blit(pygame.transform.scale(self.running_images[self.frame_run // 8],
                                               (Const.DINO_WIDTH, Const.DINO_HEIGHT)), (self.x, self.y))
            self.frame_run += 1

        # JUMPING
        if self.is_jumping or self.y < Const.DINO_OFFSET_TOP:
            window.blit(pygame.transform.scale(self.running_images[0],
                                               (Const.DINO_WIDTH, Const.DINO_HEIGHT)), (self.x, self.y))

        # DUCKING
        if self.y == Const.DINO_OFFSET_TOP_DUCKING:
            if self.frame_duck >= 16:
                self.frame_duck = 0
            window.blit(pygame.transform.scale(self.ducking_images[self.frame_duck // 8],
                                               (Const.DINO_WIDTH_DUCKING, Const.DINO_HEIGHT_DUCKING)), (self.x, self.y))
            self.frame_duck += 1


        # Draw rect on agent
        pygame.draw.rect(window, self.color, (self.x, self.y, self.w, self.h), 1)

        # Reset
        self.is_running = False
        self.is_ducking = False

    def update(self, action):

        # update agent state and score based on given action

        # RUN
        if action == 0 and not self.is_jumping:
            self.score += 1
            self.y = Const.DINO_OFFSET_TOP
            self.w = Const.DINO_WIDTH
            self.h = Const.DINO_HEIGHT

        # JUMP
        if action == 2:
            self.is_jumping = True

        # DUCK
        if action == 1 and not self.is_jumping:
            self.score += 0.5                               # Penalty for unnecessary duck
            self.y = Const.DINO_OFFSET_TOP_DUCKING
            self.w = Const.DINO_WIDTH_DUCKING
            self.h = Const.DINO_HEIGHT_DUCKING

        if self.is_jumping:
            self.score += 0.05                              # Penalty for unnecessary jump
            self.w = Const.DINO_WIDTH
            self.h = Const.DINO_HEIGHT

            if self.jump_count_running_var >= -self.jump_count_const:
                negative = 1

                if self.jump_count_running_var < 0:
                    negative = -1

                self.y -= numpy.power(numpy.abs(self.jump_count_running_var), 2) * 0.5 * negative
                self.jump_count_running_var -= 0.7

            else:
                self.is_jumping = False
                self.jump_count_running_var = self.jump_count_const

                self.y = Const.DINO_OFFSET_TOP
                self.w = Const.DINO_WIDTH
                self.h = Const.DINO_HEIGHT


    def observe(self, speed, obstacles):

        obstacle_distance_x = 1
        obstacle_distance_y = 1

        if not len(obstacles) == 0:
            obstacle_distance_x = (obstacles[0].x + self.x *(-1)) / Const.DISTANCE_OFFSET_X
            obstacle_distance_y = (obstacles[0].y) / Const.DISTANCE_OFFSET_Y

        dino_y = self.y / Const.DISTANCE_OFFSET_Y
        dino_y_vel = self.jump_count_running_var / 30
        game_speed = speed / 200

        observation = numpy.array([obstacle_distance_x, obstacle_distance_y, dino_y, dino_y_vel, game_speed])

        return observation