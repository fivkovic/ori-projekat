import pygame
from game import Game
from ga import GeneticAlgorithm
import pickle

class AIGame(Game):
    def __init__(self, first_init=True):
        super().__init__()

        self.scores_for_average = []

        if first_init == True:
            self.enable_game_rendering = False
            self.high_score = []

            self.population_size = 1000
            self.generation = 1
            self.agents_list, self.active_agents = GeneticAlgorithm.initialize(self.population_size)

    def step(self):

        def check_for_collision(agent):
            if len(self.obstacles) != 0:
                has_collided = self.obstacles[0].collide(agent)
                if has_collided:
                    self.active_agents.pop(self.active_agents.index(agent))
                    if len(self.active_agents) <= 20:
                        self.scores_for_average.append(self.game_score)

        if self.enable_game_rendering:
            self.clock.tick(self.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                #with open(f'network.pickle', 'wb') as f:
                #    pickle.dump(self.agents_list[0].neural_network, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.close()
            key = pygame.key.get_pressed()

            if key[pygame.K_SPACE]:
                self.enable_game_rendering = not self.enable_game_rendering
            if key[pygame.K_ESCAPE]:
                self.close()

        self.increment_object_counters()
        self.increment_velocity()

        for agent in self.active_agents:
            check_for_collision(agent)

        for agent in self.active_agents:
            check_for_collision(agent)
            observation = agent.observe(self.velocity, self.obstacles)                  # Observe
            action = agent.neural_network.select_action(observation)                    # Perform action
            agent.update(action)                                                        # Update

        for agent in self.active_agents:
            check_for_collision(agent)


        if len(self.active_agents) == 0:

            print("=============================================================================")
            print("GENERATION: " + str(self.generation))
            print("Average [20]: " + str(sum(self.scores_for_average) / len(self.scores_for_average)))
            print("Max     [20]: " + str(max(self.scores_for_average)))

            self.generation += 1
            self.high_score.append(self.game_score)

            pool = GeneticAlgorithm.perform_selection(self.agents_list)

            self.agents_list, self.active_agents = GeneticAlgorithm.reproduce(self.population_size, pool)

            self.reset_game(complete_init_flag = False)

        self.add_obstacle()
        self.update_obstacles()

        if self.enable_game_rendering:
            self.add_clouds()
            self.add_ground()
            self.update_ground()
            self.update_clouds()

    def render(self):

        font = pygame.font.SysFont('Helvetica', 16)

        if self.enable_game_rendering:
            self.window.fill(self.window_color)

            for i in self.grounds:
                i.draw(self.window)
            for i in self.clouds:
                i.draw(self.window)
            for i in self.obstacles:
                i.draw(self.window)
            for i in self.active_agents:
                i.draw(self.window)

            g_c = font.render('Score: ' + str(self.game_score), True, (0,0,0))
            self.window.blit(g_c, (580,10))
            d_a = font.render('Agents alive: ' + str(len(self.active_agents)), True, (0,0,0))
            self.window.blit(d_a, (10,10))
            gen = font.render('GENERATION: ' + str(self.generation), True, (0,0,0))
            self.window.blit(gen, (200,10))

            if len(self.high_score) > 0:
                high = font.render('Highscore: ' + str(max(self.high_score)), True, (0,0,0))
                self.window.blit(high, (370,10))

        elif self.speed_cnt % 40 == 0:
            a = 135
            b = 275
            self.window.fill(self.window_color)

            bla = font.render('Press SPACE to toggle rendering', True, (0,0,0))
            self.window.blit(bla, (b,a))

            gen = font.render('GENERATION: ' + str(self.generation), True, (0,0,0))
            self.window.blit(gen, (b,a+20))

            if len(self.active_agents) > 0:
                dino_number = font.render('Agents alive: ' + str(len(self.active_agents)), True, (0,0,0))
                self.window.blit(dino_number, (b,a+40))

            g_c = font.render('Score: ' + str(self.game_score), True, (0,0,0))
            self.window.blit(g_c, (b,a+60))

            if len(self.high_score) > 0:
                high = font.render('Highscore: ' + str(max(self.high_score)), True, (0,0,0))
                self.window.blit(high, (b,a+80))

        pygame.display.update()

    def reset_game(self, complete_init_flag):
        self.__init__(first_init = complete_init_flag)

if __name__ == '__main__':
    env = AIGame()
    while True:
        env.step()
        env.render()
    env.close()