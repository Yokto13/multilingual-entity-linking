from data_processors.tokens.tokens_cutter import TokensCutter
from data_processors.tokens.mewsli.tokens_iterator import (
    MewsliTokensIterator,
    ContextMewsliTokensIterator
)


class MewsliTokensIteratorFinetuning(MewsliTokensIterator):
    def __init__(
        self,
        mewsli_tsv_path,
        tokenizer,
        max_mention_tokens=-1,
        expected_size=64,
        treat_qids_as_ints=True,
        mention_token="[M]",
    ):  
        self.mention_token = mention_token
        tokenizer.add_special_tokens({"cls_token": mention_token})
        super().__init__(
            mewsli_tsv_path,
            tokenizer,
            max_mention_tokens,
            True,
            expected_size,
            treat_qids_as_ints,
        )
        assert mention_token in tokenizer.get_vocab()

    def _get_iterator(self):
        return ContextMewsliTokensIteratorFinetuning(self)


class ContextMewsliTokensIteratorFinetuning(ContextMewsliTokensIterator):
    def __init__(self, mewsli_tokens_itarator) -> None:
        super().__init__(mewsli_tokens_itarator)

    def _get_tokens_from_text_and_row(self, text, row):
        mention_slice = self.mewsli_tokens_itarator.get_mention_slice_from_row(row)
        text, mention_slice = self._add_class_token(text, mention_slice)
        token_cutter = TokensCutter(
            text,
            self.mewsli_tokens_itarator.tokenizer,
            self.mewsli_tokens_itarator.expected_size,
        )
        return token_cutter.cut_mention_with_context(mention_slice)

    def _add_class_token(self, text, mention_slice):
        new_text = f"{text[:mention_slice.start]}{self.mewsli_tokens_itarator.mention_token} {text[mention_slice]} {self.mewsli_tokens_itarator.mention_token}{text[mention_slice.stop:]}"
        new_slice = slice(
            mention_slice.start + len(self.mewsli_tokens_itarator.mention_token) + 1,
            mention_slice.stop + len(self.mewsli_tokens_itarator.mention_token) + 1,
        )
        return new_text, new_slice
