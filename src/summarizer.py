import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding

class Summarizer:
  def __init__(self, device, min_length: int, max_length: int, summary_max_length: float, num_beams: int, length_penalty: float):
    self.device = device
    self.model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-6-6").to(self.device)
    self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6", additional_special_tokens=[f'(MD{n})' for n in range(100)])
    self.model.resize_token_embeddings(len(self.tokenizer))
    self.min_length = min_length
    self.max_length = max_length
    self.summary_max_length = summary_max_length
    self.num_beams = num_beams
    self.length_penalty = length_penalty
    self.model.config.max_length = max_length
    print(self.model.config)

  def encode(self, text: str):
    return self.tokenizer.encode_plus(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

  def decode(self, value):
    return self.tokenizer.decode(value, clean_up_tokenization_spaces=True).replace("<pad>", "").replace("<s>", "").replace("</s>", "")

  def train(self, epoch: int, loader, optimizer):
    self.model.train()
    for step, data in enumerate(loader, 0):
      y = data['target_ids'].to(self.device, dtype=torch.long)
      y_ids = y[:, :-1].contiguous()
      lm_labels = y[:, 1:].clone().detach()
      lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
      ids = data['source_ids'].to(self.device, dtype=torch.long)
      mask = data['source_mask'].to(self.device, dtype=torch.long)

      outputs = self.model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=lm_labels)
      loss = outputs.loss

      if step % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  def generate(self, ids, mask):
    return self.model.generate(
      input_ids=ids,
      min_length=self.min_length,
      max_length=self.summary_max_length,
      attention_mask=mask,
      early_stopping=True,
      num_beams=self.num_beams,
      length_penalty=self.length_penalty
    )

  def validate(self, loader):
    self.model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
      for step, data in enumerate(loader, 0):
        y = data['target_ids'].to(self.device, dtype=torch.long)
        ids = data['source_ids'].to(self.device, dtype=torch.long)
        mask = data['source_mask'].to(self.device, dtype=torch.long)

        generated_ids = self.generate(ids, mask)
        preds = [self.decode(generated_id) for generated_id in generated_ids]
        target = [self.decode(t) for t in y]
        
        if step % 10 == 0:
          print(f"Completed: {step}")

        predictions.extend(preds)
        actuals.extend(target)
    return predictions, actuals

  def summarize(self, encoding: BatchEncoding):
    ids = encoding['input_ids'].to(self.device, dtype=torch.long)
    mask = encoding['attention_mask'].to(self.device, dtype=torch.long)

    self.model.eval()
    beam_outputs = self.generate(ids, mask)

    results = []
    for beam_output in beam_outputs:
      results.append(self.decode(beam_output))
    return results