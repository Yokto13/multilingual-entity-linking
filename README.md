# Multilingual Entity Linking Using Dense Retrieval

This repository contains source code experiments from the thesis [Multilingual Entity Linking Using Dense Retrieval](https://arxiv.org/abs/2406.16892).

## Getting the Data

To run the finetuning pipeline, the [DaMuEL](https://arxiv.org/abs/2306.09288) dataset is need.
It can be downloaded with scripts from `src/scripts/damuel`.
The resulting data can be than tokenized with `src/generate_descriptions_tokens.sh` and `src/generate_links_tokens.sh`.
To run the actual pipeline, consult the `src/run_finetuning.sh` and `src/run_finetuning_round.sh` scripts.

The finetuning also periodically evaluates on the [Mewsli-9](https://aclanthology.org/2020.emnlp-main.630/) dataset which can be downloaded by
following instruction from its official [repository](https://github.com/google-research/google-research/blob/master/dense_representations_for_entity_retrieval/mel/mewsli-9.md#get-mewsli-9-dataset).

## run_action.py

The majority of experiments can be run using the `run_action.py` script.

The script expects multiple arguments.
The first argument is the action to be performed.
The rest are then passed directly to the action.

The most useful actions are:

**embs** - Generate embeddings for the given dataset.

**token_index** - Built and save the index as described in the thesis.

**generate** - Generate the dataset for fine-tuning.

**train** - Train the model.

**evaluate** - Evaluate the model.

**copy** - Short cut for copying and extracting tokens to the working directory.

**at_lemmas** - Alias table with lemmatization. In development.

**at_one** - OLAT (alias table evaluated for just one language).

**at_all** - Standard alias table.

**string_similarity** - String similarity alias table as described in the thesis.

**recalls** - Calculate recalls from directories with embeddings, used inside the `evaluate` action.

**tokens_mewsli** - Generate tokens for the Mewsli-9 dataset.

**tokens_descriptions** - Generate tokens for descriptions from the DaMuEL dataset.

**tokens_links** - Generate tokens for links from the DaMuEL dataset.

The `src` directory also contains bash scripts that wrap some of the actions to make their usage easier.

Note, that I use [wandb](https://wandb.ai/home) to log our experiments. 
In either to run the code you have to log in to your wandb account or disable it by removing it from `run_action.py`
and grepping and removing all ocurrences of wandb.log(metric) from the source codes.

In accordance with the Code of Ethics of the Faculty of Mathematics and Physics of Charles University I would like to acknowledge that
part of the codebase was written in [Visual Studio
Code](https://code.visualstudio.com/) with [Github Copilot
extension](https://code.visualstudio.com/docs/copilot/overview) enabled.
