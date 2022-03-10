# Article Rewriter

Create an environment with dependencies specified in environment.yml:
```sh
conda env create -f environment.yml
```

Activate the new environment:
```sh
conda activate article-rewriter
```

Note: all Medium articles collected inside `data/` was collected using a custom Chrome extension.

Add your raw data to annotate inside `data/summarizer/markdown_data.csv` and `data/paraphraser/markdown_data.csv`.

Prepare data:
```sh
python src/prepare_data_paraphraser.py
python src/prepare_data_summarizer.py
```

Add the markdown back to the `target_without_markdown` in `data/summarizer/to_annotate_data.csv` and `data/paraphraser/to_annotate_data.csv` under a new column `target_with_markdown`.

Store these new files as `data/summarizer/annotated.csv` and `data/paraphraser/annotated.csv`.

Fine-tune data:
```sh
python src/finetune_paraphraser.py
python src/finetune_summarizer.py
```

Resulting models will be stored under `models/summarizer` and `models/paraphraser`.

Evaluate models:
```sh
python src/evaluate_paraphraser.py
python src/evaluate_summarizer.py
```

Rewrite articles (summarize then paraphrase and add non-paragraph parts back):
```sh
python src/rewrite.py
```

Deactivate an active environment:
```sh
conda deactivate
```