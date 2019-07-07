import pygame
import random
import os

class Environment():
    def __init__(self):
        self.path = os.getcwd() + '/sprites/'

    def update(self):
        self.x -= self.vel

    def draw(self, window):
        window.blit(self.img, (self.x, self.y))


class Cloud(Environment):

    def __init__(self):
        super().__init__()
        self.x   = 830
        self.y   = random.randrange(50,180)
        self.vel = 3
        self.img = pygame.image.load(self.path + 'cloud.png')
        self.width = self.img.get_width()

class Ground(Environment):

    def __init__(self, x, vel):
        super().__init__()
        self.x = x
        self.y = 277
        self.vel = vel
        self.img = pygame.image.load(self.path + 'ground.png')
        self.width = self.img.get_width()


class Obstacle():
    def __init__(self, vel):
        self.x = 830
        self.vel = vel
        self.path = os.getcwd() + '/sprites/'

    def update(self):
        self.x -= self.vel

    def draw(self, window):
        pass

    def draw_hitbox(self, window):
        pygame.draw.rect(window, self.color,
                     (self.x,self.y,self.width,self.height),1)

    def collide(self, dino):
        if pygame.Rect(self.x, self.y, self.width, self.height).colliderect(
                pygame.Rect(dino.x, dino.y, dino.w, dino.h)):
            return True
        else:
            return False

class Cactus(Obstacle):

    def __init__(self, vel):
        super().__init__(vel)

        self.img  = [pygame.image.load(self.path + 'cactus_' + str(i) + '.png') for i in range(0,16)]
        self.r = random.randrange(0,16)
        self.curr_cactus = self.img[self.r]

        self.color = (231,76,60)
        self.width  = int(self.curr_cactus.get_width()  * 2/3)
        self.height = int(self.curr_cactus.get_height() * 2/3)
        self.y      = 296 - self.height

    def draw(self, window):
        window.blit(pygame.transform.scale(self.curr_cactus, (self.width, self.height)), (self.x, self.y))

        self.draw_hitbox(window)

class Ptera(Obstacle):

    def __init__(self, vel):
        super().__init__(vel)

        self.img     = [pygame.image.load(self.path + 'ptera_0.png'),
                        pygame.image.load(self.path + 'ptera_1.png')]
        self.frame   = 0

        self.y = random.randrange(90, 220)
        self.color = (44,62,80)
        self.height  = int(self.img[0].get_height() * 2/3)
        self.width   = int(self.img[0].get_width() * 2/3)

    def draw(self, window):
        if self.frame >= 16:
                self.frame = 0
        window.blit(pygame.transform.scale(self.img[self.frame // 8], (61,53)), (self.x, self.y))
        self.frame += 1

        self.draw_hitbox(window)