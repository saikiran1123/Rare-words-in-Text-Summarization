# rare-words-summarization

Text summarization with a Transformer that handles rare words. It can run with
or without a pointer-generator copy mechanism, controlled by one flag in the
config.

Based on the paper "Rare words in text summarization" (Morozovskii & Ramanna, 2023).

## What it does

- First it picks the most useful sentences from the article (extractive step).
- Then a Transformer writes a summary from those sentences (abstractive step).
- The encoder attention gives more weight to rare words.
- If the pointer is on, the model can also copy words straight from the article.

## Files

- `main.py` — the command to run everything (prepare, train, evaluate).
- `config.yaml` — all the settings (model size, batch size, epochs, pointer on/off, etc.).
- `requirements.txt` — the Python packages needed.
- `summarizer/__init__.py` — loads the config and sets random seeds.
- `summarizer/data.py` — picks important sentences, builds the vocabulary, and prepares the data for the model.
- `summarizer/model.py` — the Transformer, the rare-word attention, and the pointer-generator.
- `summarizer/train.py` — the training loop, loss, and saving the best model.
- `summarizer/infer.py` — generates summaries (beam search) and scores them with ROUGE.
- `notebooks/` — the two original notebooks, kept for reference.

## Architecture

- **Two stages:** an extractive step picks the important sentences, then an abstractive Transformer writes the summary from them.
- **Extractive step:** it scores 3-word phrases by how rare they are, ranks sentences by their average score, and keeps the top sentences up to a token budget (default 400).
- **Sentence order:** the kept sentences are put back in their original order before going to the model.
- **Transformer:** a from-scratch encoder–decoder (8 layers, 8 heads, model size 256), trained from scratch with no pre-trained weights.
- **Rare-word attention:** the encoder gives more attention weight to rare words and less to common ones, so important uncommon words are not ignored.
- **Pointer-generator:** at each step the model chooses between writing a new word from its vocabulary or copying a word straight from the article.
- **Why copying helps:** it lets the model handle names, places, and other out-of-vocabulary words by pulling them from the source text.
- **The gate (p_gen):** a small learned value decides how much to generate vs how much to copy at each step.
- **Decoding:** summaries are produced with beam search and a length penalty.
- **One flag:** `use_pointer` turns the copy mechanism on or off, which is how you compare the two versions.

## Setup

```
pip install -r requirements.txt
```

NLTK data downloads on the first run.

## How to run

```
python main.py prepare  --config config.yaml
python main.py train    --config config.yaml
python main.py evaluate --config config.yaml
```

To turn the pointer on or off, change `use_pointer` in `config.yaml`.
- `use_pointer: true`  saves `best_model_pg.pt`
- `use_pointer: false` saves `best_model_nopg.pt`

## Config

All settings are in `config.yaml`. The main ones are `use_pointer`,
`d_model`, `n_layers`, `n_heads`, `batch_size`, `epochs`, `lr`, `beam_size`,
and `subsample_size` (set to a number to train on fewer rows).

## Dataset

CNN/DailyMail, loaded from HuggingFace. By default it uses the full training set
(~287k rows). Set `subsample_size` to a number for a smaller run.

## Evaluation

Reports ROUGE-1, ROUGE-2 and ROUGE-L on the test set using beam search.
Train with the pointer on and off to compare the two. No scores are included
in this repo; you get them by running it yourself.

## Notes

A few things differ from the paper: it uses the Adam optimizer, an NLL loss,
and a single 400-token input size. Training is from scratch with no pre-trained
model, so results will be lower than large pre-trained models.

## License

MIT.