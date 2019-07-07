import time
import cv2
import numpy
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from PIL import ImageGrab

DEATH_PENALTY = -10000
JUMP_REWARD = 100
UNNECESSARY_JUMP_PENALTY = -200
LIVING_REWARD = 5

class Game:

    def __init__(self):

        self.observation_space = 2 ** 25      # 20 pixels + 4 score + jump state
        self.action_space = 2
        self.jump_key = 2 ** 19

        cli_switches = webdriver.ChromeOptions()
        cli_switches.add_argument('--disable-infobars')
        self._chrome = webdriver.Chrome(options=cli_switches)

        self._chrome.set_window_position(-10,0)
        self._chrome.set_window_size(670, 240)

        self._chrome.get('chrome://dino/')
        time.sleep(1)

        self._chrome.execute_script('Runner.config.ACCELERATION=0')

    # =====================================================================
    # BROWSER CONTROL
    # =====================================================================

    def start(self):
        self.jump()

    def restart(self):
        script = 'Runner.instance_.restart()'
        self._chrome.execute_script(script)
        time.sleep(0.5)
        return self.get_state()

    def quit(self):
        self._chrome.quit()

    def jump(self):
        self._chrome.find_element_by_tag_name('body').send_keys(Keys.SPACE)

    def get_score(self):
        script = 'return Runner.instance_.distanceMeter.digits'
        digits = self._chrome.execute_script(script)
        score = ''.join(digits)

        return int(score)

    def is_currently_jumping(self):
        script = 'return Runner.instance_.tRex.jumping'
        return self._chrome.execute_script(script)

    def is_game_over(self):
        script = 'return Runner.instance_.crashed'
        return self._chrome.execute_script(script)

    # =====================================================================
    # ENVIRONMENT CONTROL
    # =====================================================================

    def perform_action(self, selected_action):

        if selected_action == 1:
            self.jump()

        current_state = self.get_state()
        game_over = self.is_game_over()

        if game_over:
            reward = DEATH_PENALTY
        elif (current_state % 2 == 1):               # JUMP
            if current_state >= self.jump_key:
                reward = JUMP_REWARD
            else:                                    # penalty - unnecessary jump
                reward = UNNECESSARY_JUMP_PENALTY
        else:
            reward = LIVING_REWARD

        return current_state, reward, game_over

    def get_state(self):

        bb_x = 112
        bb_y = 111
        bb_w = 492
        bb_h = 108

        image = ImageGrab.grab(bbox=(bb_x, bb_y, bb_x + bb_w, bb_y + bb_h))
        image_array = numpy.array(image)

        converted_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        converted_image = cv2.threshold(converted_image, 127, 255, cv2.THRESH_BINARY)[1]

        if numpy.median(converted_image[0, :]) == 0:
            converted_image = numpy.invert(converted_image)

        state = self._detect_obstacles(converted_image).tolist()
        state = [str(int(x)) for x in state]

        state_number = int(''.join(state), base=2)                              # bool to int
        state_number = state_number << 1 | int(self.is_currently_jumping())     # add info if currently jumping

        return state_number

    @staticmethod
    def _detect_obstacles(converted_image, new_length=20):
        _, ix = converted_image.shape
        step_size = ix // new_length
        res = numpy.zeros(new_length, dtype=numpy.uint8)

        for i in range(new_length):
            a = i * step_size
            b = a + step_size
            if not converted_image[:, a:b].all():
                res[i] = 1

        return res