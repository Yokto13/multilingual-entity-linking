from pathlib import Path

import pandas as pd

from data_processors.tokens.mention_qid_pair import MentionQidPair
from data_processors.tokens.tokens_cutter import TokensCutter


class MewsliTokensIterator:
    def __init__(
        self,
        mewsli_tsv_path,
        tokenizer,
        use_context=True,
        max_mention_tokens=-1,
        expected_size=64,
        treat_qids_as_ints=True,
    ):
        self.mewsli_tsv_path = self._validate_and_get_path(mewsli_tsv_path)
        self.use_context = use_context
        self.expected_size = self._validate_and_get_expected_size(expected_size)

        self.max_mention_tokens = max_mention_tokens
        self.tokenizer = tokenizer
        self.treat_qids_as_ints = treat_qids_as_ints
        self.qid_parser = self._get_qid_parser()

        self.data_df = pd.read_csv(mewsli_tsv_path, sep="\t")

    def _validate_and_get_path(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        assert path.exists(), f"Path {path} does not exist."
        return path

    def _validate_and_get_expected_size(self, expected_size):
        if self.use_context:
            assert expected_size > 0, "expected_size must be greater than 0."
        return expected_size

    def __iter__(self):
        iterator = self._get_iterator()
        for x in iterator:
            yield x

    def _get_iterator(self):
        if self.use_context:
            return ContextMewsliTokensIterator(self)
        else:
            return NoContextMewsliTokensIterator(self)

    def _qid_to_int(self, qid):
        return int(qid[1:])

    def _qid_to_str(self, qid):
        return qid

    def _get_qid_parser(self):
        if self.treat_qids_as_ints:
            return self._qid_to_int
        else:
            return self._qid_to_str

    def get_mention_slice_from_row(self, row):
        mention_start = row.position
        mention_end = row.position + row.length
        return slice(mention_start, mention_end)


class ContextMewsliTokensIterator:
    def __init__(self, mewsli_tokens_itarator) -> None:
        self.mewsli_tokens_itarator = mewsli_tokens_itarator

    def __iter__(self):
        for row in self.mewsli_tokens_itarator.data_df.itertuples():
            qid = self.mewsli_tokens_itarator.qid_parser(row.qid)
            with open(
                self.mewsli_tokens_itarator.mewsli_tsv_path.parent / "text" / row.docid,
                "r",
            ) as f:
                text = f.read()

                try:
                    toks = self._get_tokens_from_text_and_row(text, row)
                    yield MentionQidPair(toks, qid)
                except TypeError as e:
                    print("Error in cutting tokens because of wrong mention slice.")
                    print(e)
                    print(
                        self.mewsli_tokens_itarator.get_mention_slice_from_row(row),
                        f"START{text[self.mewsli_tokens_itarator.get_mention_slice_from_row(row)]}END",
                    )

    def _get_tokens_from_text_and_row(self, text, row):
        mention_slice = self.mewsli_tokens_itarator.get_mention_slice_from_row(row)
        token_cutter = TokensCutter(
            text,
            self.mewsli_tokens_itarator.tokenizer,
            self.mewsli_tokens_itarator.expected_size,
        )
        return token_cutter.cut_mention_with_context(mention_slice)


class NoContextMewsliTokensIterator:
    def __init__(self, mewsli_tokens_iterator):
        self.mewsli_tokens_iterator = mewsli_tokens_iterator

    def __iter__(self):
        for row in self.mewsli_tokens_iterator.data_df.itertuples():
            mention = row.mention
            qid = self.mewsli_tokens_iterator.qid_parser(row.qid)

            yield MentionQidPair(
                self.mewsli_tokens_iterator.tokenizer(
                    mention,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.mewsli_tokens_iterator.expected_size,
                ),
                qid,
            )
