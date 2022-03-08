from config_summarizer import Config
from summarizer import Summarizer
from finetune import finetune

config = Config()
summarizer = Summarizer(config)
finetune(config, summarizer)
