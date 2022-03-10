import numpy as np
import torch

class Config:
  seed = 42
  learning_rate = 1e-4
  train_batch_size = 2
  val_batch_size = 2
  train_epochs = 3
  min_length = 0
  max_length = 512
  summary_max_length = 142
  num_beams = 4
  length_penalty = 0.1
  rewriter_name = "summarizer"

  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    torch.backends.cudnn.deterministic = True