# FineWeb-Edu

- **Source:** [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (subset `sample-10BT`)
- **Shape:** real educational web pages, **one per document**, separated by `<|endoftext|>`.
  The full corpus is ~1.3T tokens / ~3.5B documents / ~5.84 TB; we stream a slice.
- **Used by:** the "real knowledge" line (`fineweb_edu/*`, `tokenizer/*`, `chinchilla/*`).

### Examples

```
Cori spezzati (kor' ee spetz ah' tee), Italian for "separated choirs," is the term
used to describe a musical composition that uses spatial effects to emphasize the
interplay between its various voices. Typically, this means placing two or more
choirs or groups of instruments in various places around a performance space. ...
```

```
CTComms sends on average 2 million emails monthly on behalf of over 125 different
charities and not for profits.

Take the complexity of technology and stir in the complexity of the legal system
and what do you get? Software licenses! If you've ever attempted to read one you
know how true this is ...
```
