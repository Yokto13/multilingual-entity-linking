# Multilingual Entity Linking Using Dense Retrieval

This repository contains source code experiments from the thesis "Multilingual Entity Linking Using Dense Retrieval".

To run the finetuning pipeline, you need to download DaMuEL which can be done with scripts
from `src/scripts/damuel`.
The resulting data can be than tokenized with `src/generate_descriptions_tokens.sh' and `src/generate_links_tokens.sh`.
To run the actual pipeline, consult the `src/run_finetuning.sh` and `src/run_finetuning_round.sh` scripts.
The finetuning also periodically evaluates on the Mewsli-9 dataset which can be downloaded by
following instruction from its official [repository](https://github.com/google-research/google-research/blob/master/dense_representations_for_entity_retrieval/mel/mewsli-9.md#get-mewsli-9-dataset).