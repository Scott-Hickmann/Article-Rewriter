from config_paraphraser import Config
from paraphraser import Paraphraser
from finetune import finetune

config = Config()
paraphraser = Paraphraser(config)
finetune(config, paraphraser)
