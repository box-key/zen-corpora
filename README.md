Zen-corpora
-----------

Description
-----------
Zen-corpora provides two main funcitonalities:
- A memory efficient way to store unique sentences in corpus.
- Beam text search with RNN model in PyTorch.

Installation
------------
This module requires Python 3.7+. Please install it by running:
```bash
pip install zen-corpora
```

Why Zen-corpora?
----------------
Think about how Python stores the corpus below:
```python
corpus = [['I', 'have', 'a', 'pen'],
          ['I', 'have', 'a', 'dog'],
          ['I', 'have', 'a', 'cat'],
          ['I', 'have', 'a', 'tie']]
```
It stores each sentence separately, but it's wasting the memory by storing "I have a " 4 times.

Zen-corpora solves this problem by storing sentences in a corpus-level trie. For example, the corpus above will be stored as
```bash
<<<<<<< HEAD
├─ I ─ have ─ a
      	        ├─ pen
		├─ dog
                ├─ cat
	        └─ tie
=======
├── I ── have ── a 
      	         ├── pen
		 ├── dog
                 ├── cat 
	         └── tie
>>>>>>> 61d1de816812db55339d85c6297dfdcfbf688a51
```
In this way, we can save lots of memory space and sentence search can be a lot faster!

Zen-corpora provides Python API to easily construct and interact with a corpus trie. See the following example:
```python
>>> import zencorpora
>>> from zencorpora.corpustrie import CorpusTrie
>>> corpus = [['I', 'have', 'a', 'pen'],
...           ['I', 'have', 'a', 'dog'],
...           ['I', 'have', 'a', 'cat'],
...           ['I', 'have', 'a', 'tie']]
>>> trie = CorpusTrie(corpus=corpus)
>>> print(len(trie))
7
>>> print(['I', 'have', 'a', 'pen'] in trie)
True
>>> print(['I', 'have', 'a', 'sen'] in trie)
False
>>> trie.insert(['I', 'have', 'a', 'book'])
>>> print(['I', 'have', 'a', 'book'] in trie)
True
>>> print(trie.remove(['I', 'have', 'a', 'book']))
1
>>> print(['I', 'have', 'a', 'book'] in trie)
False
>>> print(trie.remove(['I', 'have', 'a', 'caw']))
-1
>>> print(trie.make_list())
[['i', 'have', 'a', 'pen'], ['i', 'have', 'a', 'dog'], ['i', 'have', 'a', 'cat'], ['i', 'have', 'a', 'tie']]
```

Left-to-Right Beam Text Search
------------------------------
As shown in SmartReply paper by [Kannan et al. (2016)](https://www.kdd.org/kdd2016/papers/files/Paper_1069.pdf), corpus trie can be used to perform left-to-right beam search using RNN model.
A model encodes input text, then it computes the probability of each pre-defined sentence in the searching space given the encoded input.
However, this process is exhaustive. What if we have 1 million sentences in the search space? Without beam search, a RNN model processes 1 million sentences.
Thus, the authors used the corpus trie to perform a beam search for their pre-defined sentences.
The idea is simple, it starts search from the root of the trie. Then, it only retains beam width number of probable sentences at each level.

Zen-corpora provides a class to enable beam search. See the example below.
```python
>>> import torch.nn as nn
>>> import torch
>>> import os
>>> from zencorpora import SearchSpace
>>> corpus_path = os.path.join('data', 'search_space.csv')
>>> data = ... # assume data contains torchtext Field, encoder and decoder
>>> space = SearchSpace(
...    src_field=data.input_field,
...    trg_field=data.output_field,
...    encoder=data.model.encoder,
...    decoder=data.model.decoder,
...    corpus_path=corpus_path,
...    hide_progress=False,
...    score_function=nn.functional.log_softmax,
...    device=torch.device('cpu'),
... ) # you can hide a progress bar by setting hide_progress = False
Construct Corpus Trie: 100%|████████████████████████████████████████| 34105/34105 [00:01<00:00, 21732.69 sentence/s]
>>> src = ['this', 'is', 'test']
>>> result = space.beam_search(src, 2)
>>> print(len(result))
2
>>> print(result)
[('is this test?', 1.0), ('this is test!', 1.0)]
>>> result = space.beam_search(src, 100)
>>> print(len(result))
100
```

License
-------
This project is licensed under Apache 2.0.
