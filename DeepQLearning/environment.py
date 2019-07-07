from PIL import Image
from io import BytesIO
from server import WebsocketServer
import base64, json, re, time, threading
import multiprocessing
import numpy as np

class Action:
    UP = 0
    DOWN = 1
    FORWARD = 2

class Environment:

    actions = {Action.UP:'UP', Action.FORWARD:'FORTH', Action.DOWN:'DOWN'}

    def __init__(self, host, port, debug=False):
        self.debug = debug
        self.queue = multiprocessing.Queue()
        self.game_client = None
        self.server = WebsocketServer(port, host=host)
        self.server.set_fn_new_client(self.new_client)
        self.server.set_fn_message_received(self.new_message)
        print("\nGame can be connected (press F5 in Browser)")
        thread = threading.Thread(target = self.server.run_forever)
        thread.daemon = True
        thread.start()

    def new_client(self, client, server):
        if self.debug: print("GameAgent: Game just connected")
        self.game_client = client
        self.server.send_message(self.game_client, "Connection to Game Agent Established");

    def new_message(self, client, server, message):
        if self.debug: print("GameAgent: Incoming data from game")
        data = json.loads(message)
        image, crashed = data['world'], data['crashed']

        # remove data-info at the beginning of the image
        image = re.sub('data:image/png;base64,', '',image)
        # convert image from base64 decoding to np array
        image = np.array(Image.open(BytesIO(base64.b64decode(image))))
        #Image.fromarray(image).show()

        # cast to bool
        crashed = True if crashed in ['True', 'true'] else False

        self.queue.put((image, crashed))

    def start_game(self):
        while self.game_client is None:
            time.sleep(1)

        self.server.send_message(self.game_client, "START");
        time.sleep(4)
        return self.get_state(Action.FORWARD)

    def refresh_game(self):
        time.sleep(0.5)
        print("Refreshing...")
        self.server.send_message(self.game_client, "REFRESH");
        time.sleep(1)

    def do_action(self, action):

        if action != Action.FORWARD:
            self.server.send_message(self.game_client, self.actions[action]);

        time.sleep(.05)
        return self.get_state(action)

    def get_state(self, action):
        self.server.send_message(self.game_client, "STATE");

        image, crashed = self.queue.get()

        if crashed:
            reward = -100.
        else:
            if action == Action.UP:
                reward = -5.
            elif action == Action.DOWN:
                reward = -3.
            else:
                reward = 1.

        return image, reward, crashed
