# Multilingual Entity Linking Using Dense Retrieval

This repository contains source code experiments from the thesis "Multilingual Entity Linking Using Dense Retrieval".

## Getting the Data

To run the finetuning pipeline, the [DaMuEL](https://arxiv.org/abs/2306.09288) dataset is need.
It can be downloaded with scripts from `src/scripts/damuel`.
The resulting data can be than tokenized with `src/generate_descriptions_tokens.sh` and `src/generate_links_tokens.sh`.
To run the actual pipeline, consult the `src/run_finetuning.sh` and `src/run_finetuning_round.sh` scripts.

The finetuning also periodically evaluates on the Mewsli-9 dataset which can be downloaded by
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