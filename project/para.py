import ray 
from evogym_python import *


class Evosim:
    def __init__(self, config):
        self=make_env(config["env_name"], robot=config["robot"])
        self.config = config
        
def __reduce__(self):
    deserializer=Evosim
    serialized_data=(self.config)
    return (deserializer, serialized_data)

