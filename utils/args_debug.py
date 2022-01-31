class args_debug():
    def __init__(self,env_name, model,agent_id, seed, argmax, pause, shift, gif_dir, episodes, memory, text):
        self.env_name = env_name
        self.model= model
        self.agent_id = agent_id
        self.seed = seed
        self.argmax = argmax
        self.shift = shift
        self.gif_dir = gif_dir
        self.episodes = episodes
        self.memory = memory
        self.text = text
        self.pause =pause