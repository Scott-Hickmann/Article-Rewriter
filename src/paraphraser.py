import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding

class Paraphraser:
  def __init__(self, device, max_length: int, num_beams: int, num_beam_groups: int, diversity_penalty: float):
    self.device = device
    self.model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality").to(self.device)
    self.tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
    self.max_length = max_length
    self.num_beams = num_beams
    self.num_beam_groups = num_beam_groups
    self.diversity_penalty = diversity_penalty
    self.model.config.max_length = max_length

  def encode(self, context: str):
    text = "paraphrase: " + context + " </s>"
    encoding = self.tokenizer.encode_plus(text, max_length=self.max_length, padding="max_length", truncation=True, pad_to_max_length=True, return_tensors="pt")
    return encoding

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
      # print(outputs)
      loss = outputs.loss

      if step % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  def validate(self, loader):
    self.model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
      for step, data in enumerate(loader, 0):
        y = data['target_ids'].to(self.device, dtype=torch.long)
        ids = data['source_ids'].to(self.device, dtype=torch.long)
        mask = data['source_mask'].to(self.device, dtype=torch.long)

        generated_ids = self.model.generate(
          input_ids=ids,
          max_length=self.max_length,
          attention_mask=mask,
          early_stopping=True,
          num_beams=self.num_beams,
          num_return_sequences=1,
          num_beam_groups=self.num_beam_groups,
          diversity_penalty=self.diversity_penalty
        )
        preds = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
        if step % 10 == 0:
          print(f"Completed: {step}")

        predictions.extend(preds)
        actuals.extend(target)
    return predictions, actuals

  def paraphrase(self, encoding: BatchEncoding):
    ids = encoding['input_ids'].to(self.device, dtype=torch.long)
    mask = encoding['attention_mask'].to(self.device, dtype=torch.long)

    self.model.eval()
    beam_outputs = self.model.generate(
      input_ids=ids,
      max_length=self.max_length,
      attention_mask=mask,
      early_stopping=True,
      num_beams=self.num_beams,
      num_return_sequences=1,
      num_beam_groups=self.num_beam_groups,
      diversity_penalty=self.diversity_penalty
    )

    results = []
    for beam_output in beam_outputs:
      sent = self.tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
      results.append(sent.replace("paraphrasedoutput: ", ""))
    return results