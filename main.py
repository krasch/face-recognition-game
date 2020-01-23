import logging.config

from game.game import game_loop

FORMAT = '%(asctime)s.%(msecs)03d %(name)s %(message)s'
logging.basicConfig(level="INFO", format=FORMAT, datefmt='%H:%M:%S')
game_loop()
