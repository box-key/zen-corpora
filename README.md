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
corpus = ['I', 'have', 'a', 'pen',
          'I', 'have', 'a', 'dog',
          'I', 'have', 'a', 'cat',
          'I', 'have', 'a', 'tie',]
```
It stores each sentence separately, but it's waste of the memory to store "I have a " 4 times.

Zen-corpora solves this problem by storing sentences in a corpus-level trie. For example, the corpus above will be stored as 
```bash
├─ I ─ have ─ a 
      	        ├─ pen
		├─ dog
                ├─ cat 
	        └─ tie
```
In this way, we can save lots of memory space and sentence search can be a lot faster!

Zen-corpora provides Python API to easily construct and interact with a corpus trie. See the following example:
```python
from zencorpora import CorpusTrie


corpus = [['I', 'have', 'a', 'pen'],
          ['I', 'have', 'a', 'dog'],
          ['I', 'have', 'a', 'cat'],
          ['I', 'have', 'a', 'tie']]
# construct trie
trie = CorpusTrie(corpus=corpus)
# returns the number of tokens in the trie
print(len(trie))
>>> 7
# check if a trie contains the following sentences
print(['I', 'have', 'a', 'pen'] in trie)
>>> True
print(['I', 'have', 'a', 'sen'] in trie)
>>> False
# insert a sentence
trie.insert(['I', 'have', 'a', 'book'])
print(['I', 'have', 'a', 'book'] in trie)
>>> True
# remove a sentence, returns 1 if a sentence exists
print(trie.remove(['I', 'have', 'a', 'book']))
>>> 1
print(['I', 'have', 'a', 'book'] in trie)
>>> False
# returns -1 if sentence doesn't exist
print(trie.remove(['I', 'have', 'a', 'caw']))
>>> -1
# it returns corpus as a list
print(trie.make_list())
>>> [['i', 'have', 'a', 'pen'], ['i', 'have', 'a', 'dog'], ['i', 'have', 'a', 'cat'], ['i', 'have', 'a', 'tie']]

```

Left-to-Right Beam Text Search
------------------------------
As shown in SmartReply paper by [Kannan et al. (2016)](https://www.kdd.org/kdd2016/papers/files/Paper_1069.pdf), corpus trie can be used to perform left-to-right beam search using RNN model.
A model encodes input text, then it computes the probability of each pre-defined sentence in the searching space given the encoded input.
However, this process is exhausting. What if we have 1 million sentences in the search space? Without beam search, a RNN model processes 1 million sentences.
Thus, the authors used the corpus trie to perform a beam search for their pre-defined sentences. 
The idea is simple, it starts search from the root of the trie. Then, it only retains beam width number of probable sentences at each level.

Zen-corpora provides a class to enable beam search. See the example below.
```python
import torch.nn as nn
import torch 

from zencorpora import SearchSpace


# specify the path to corpus (right now it only accepts csv format)
corpus_path = os.path.join('data', 'search_space.csv')
# assume you already trained a gru model in PyTorch
# search space can be constructed as follows
# it shows a progress bar if you choose hide_progress = False
space = SearchSpace(
    src_field=data.input_field,
    trg_field=data.output_field,
    encoder=data.model.encoder,
    decoder=data.model.decoder,
    corpus_path=corpus_path,
    hide_progress=False,
    score_function=nn.functional.log_softmax,
    device=torch.device('cpu'),
)
>>> Construct Corpus Trie: 100%|████████████████████████████████████████| 34105/34105 [00:01<00:00, 21732.69 sentence/s]
# Let's search!
src = ['this', 'is', 'test']
result = space.beam_search(src, 2)
print(len(result))
>>> 2
# it returns text with its score (log probability in this example)
print(result)
>>> [('is this test?', 1.0), ('this is test!', 1.0)]
# expand a beam width, assume your search space has more than 100 sentences
result = space.beam_search(src, 100)
print(len(result))
>>> 100
```

License
-------
This project is licensed under Apache 2.0.
