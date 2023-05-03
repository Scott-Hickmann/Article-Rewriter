import numpy as np
import torch

class Config:
  seed = 42
  learning_rate = 1e-4
  train_batch_size = 2
  val_batch_size = 2
  train_epochs = 5
  max_length = 128
  num_beams = 3
  num_beam_groups = 3
  diversity_penalty = 0.99
  rewriter_name = "paraphraser"

  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(self.seed)
    np.random.seed(self.seed)
    torch.backends.cudnn.deterministic = True