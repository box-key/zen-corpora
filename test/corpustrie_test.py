import pytest
import os

from zencorpora.corpustrie import TrieNode, CorpusTrie
from test.loader import DataLoader


class TestTrieNode:

    def test_overrides(self):
        """ Test methods that override built-in methods """
        node = TrieNode('test')
        # check if token is correctly initialized
        assert node.token == 'test'
        # test repr returns a node's token
        assert repr(node) == 'test'
        # see if len method works
        assert len(node) == 0

    def test_add(self):
        """ Test add child method """
        node = TrieNode('test')
        node.add_child('code')
        # make sure a node is added to children
        assert repr(node.children[-1]) == 'code'
        node.add_child('!')
        node.add_child('code')
        # make sure the node stores unique children
        assert len(node) == 2
        # make sure it works in a loop
        test = TrieNode('test')
        for i in range(10):
            test.add_child('c'+str(i))
        assert len(test) == 10
        # make sure it ignores duplicates
        for i in range(10):
            test.add_child('c'+str(i))
        assert len(test) == 10

    def test_find(self):
        """ Test add child method """
        node = TrieNode('test')
        node.add_child('code')
        # make sure it returns -1 if it can't find element
        assert node.find('pytest') == -1
        node.add_child('python')
        # make sure it returns index when it finds an element
        assert node.find('python') > 0

    def test_leaf(self):
        """ Test is leaf method """
        node = TrieNode('test')
        assert node.is_leaf()
        node.add_child('code')
        assert not node.is_leaf()

    def test_get_child(self):
        """ Test get method on node's children """
        node = TrieNode('test')
        # returns -1 if it cannot find input
        assert node.find('code') == -1
        node.add_child('code')
        # returns token exists at index
        assert repr(node.get_child(0)) == 'code'

    def test_remove(self):
        """ Test remove method """
        node = TrieNode('test')
        node.add_child('code')
        node.remove('pytest')
        # make sure it doesn't remove existing element
        assert len(node) == 1
        node.add_child('pytest')
        node.remove('code')
        # make sure it removes elements correctly
        assert (len(node) == 1) and (repr(node.children[-1]) == 'pytest')
        # make sure it works in a loop
        test = TrieNode('test')
        for i in range(10):
            test.add_child('c'+str(i))
        for i in range(10):
            test.remove('c'+str(i))
        assert len(test) == 0


class TestCorpusTrie:

    def test_override(self):
        """ Test methods that override built-in methods """
        trie = CorpusTrie()
        # check if it initializes the root node
        assert repr(trie.root) == '<root>'
        # check len method
        assert len(trie) == 0
        # check if contains works
        trie.root.add_child('test')
        assert ['test'] in trie
        assert not ['test', 'it'] in trie
        # make sure initalizer constructs trie with a corpus
        sen1 = ['He', 'has', 'a', 'dog']
        sen2 = ['I', 'like', 'a', 'black', 'dog']
        sen3 = ['I', 'like', 'a', 'black', 'pen']
        corpus = [sen1, sen2, sen3]
        trie = CorpusTrie(corpus=corpus)
        assert len(trie) == 10

    def test_insert(self):
        trie = CorpusTrie(case_sensitive=False)
        sentence1 = ['This', 'is', 'a', 'test']
        # check if it inserts sentence into the tree correctly
        trie.insert(sentence1)
        assert len(trie) == 4
        sentence2 = ['This', 'is', 'a', 'dog']
        # check if it only stores unique tokens
        trie.insert(sentence2)
        assert len(trie) == 5
        # check if the trie is not case sensitive
        sentence3 = ['This', 'Is', 'a', 'DOG']
        trie.insert(sentence3)
        assert len(trie) == 5
        # check case sensitive feature
        sentence4 = ['this', 'is', 'a', 'dog']
        trie_cs = CorpusTrie(case_sensitive=True)
        trie_cs.insert(sentence1)
        trie_cs.insert(sentence2)
        trie_cs.insert(sentence3)
        trie_cs.insert(sentence4)
        assert len(trie_cs) == 12

    def test_remove(self):
        trie = CorpusTrie()
        # check it doesn't throw an error with empty tree
        # -1 indicates it cannot find a match to input
        assert trie.remove(['test', 'it']) == -1
        sen1 = ['He', 'has', 'a', 'dog']
        sen2 = ['I', 'like', 'a', 'black', 'dog']
        sen3 = ['I', 'like', 'a', 'black', 'pen']
        sen4 = ['I', 'like', 'it']
        sen5 = ['I', 'have', 'a', 'dog']
        sen6 = ['I', 'have', 'a', 'pen']
        for sen in [sen1, sen2, sen3, sen4, sen5, sen6]:
            trie.insert(sen)
        # check if it removes a complete sentence and returns 1
        assert trie.remove(sen2) == 1
        assert len(trie) == 14
        assert sen2 not in trie
        # make sure it retains a sibling of deleted sentence
        assert sen3 in trie
        # check if it removes the entire path without any children
        trie.remove(sen4)
        # the input should still exists
        assert ['I', 'like'] in trie
        trie.remove(sen3)
        # now the input shouldn't exists in trie
        assert ['I', 'like'] not in trie
        assert len(trie) == 9
        # make sure it doesn't remove incomplete sentence
        assert trie.remove(['He', 'has', 'a']) == -1
        # check again tp remove the entire path if it doesn't have any children
        # trie should contains this
        assert ['He'] in trie
        # now it should remove the whole sentence
        trie.remove(sen1)
        assert ['He'] not in trie
        assert len(trie) == 5

    def test_update(self):
        trie = CorpusTrie()
        sen1 = ['He', 'has', 'a', 'dog']
        sen2 = ['I', 'like', 'a', 'black', 'dog']
        sen3 = ['I', 'like', 'a', 'black', 'pen']
        corpus = [sen1, sen2, sen3]
        # make sure update methods insert all the sentences in corpus
        trie.update(corpus)
        assert len(trie) == 10
        # make sure trie structure is maintained
        sen4 = ['I', 'like', 'it']
        trie.insert(sen4)
        assert len(trie) == 11
        trie.update(corpus)
        assert len(trie) == 11

    def test_list_constructor(self):
        sen1 = ['He', 'has', 'a', 'dog']
        sen2 = ['I', 'like', 'a', 'black', 'dog']
        sen3 = ['I', 'like', 'a', 'black', 'pen']
        corpus = [sen1, sen2, sen3]
        trie = CorpusTrie(corpus=corpus, case_sensitive=True)
        # make sure make_list returns corpus as a list of sentence
        reverse_corpus = trie.make_list()
        assert len(reverse_corpus) == 3
        # make sure it returns an identical corpus
        for ori, rev in zip(corpus, reverse_corpus):
            assert ori == rev
        # make sure returned list is case sensitive
        trie = CorpusTrie(corpus=corpus, case_sensitive=True)
        reverse_corpus = trie.make_list()
        for ori, rev in zip(corpus, reverse_corpus):
            assert ori == rev

    def test_save_corpus(self):
        sen1 = ['He', 'has', 'a', 'dog']
        sen2 = ['I', 'like', 'a', 'black', 'dog']
        sen3 = ['I', 'like', 'a', 'black', 'pen']
        corpus = [sen1, sen2, sen3]
        trie = CorpusTrie(corpus=corpus, case_sensitive=True)
        file = 'test.txt'
        # check if it outputs a file
        trie.save_corpus(file)
        assert os.path.exists(file)
        # remove the test file
        os.remove(file)

    def test_load_corpus(self):
        # path to small corpus
        PATH_CORPUS_SAMPLE = os.path.join('test', 'data', 'space_sample.csv')
        # test load method
        trie_test = CorpusTrie(corpus_path=PATH_CORPUS_SAMPLE)
        loader = DataLoader(small_corpus=True)
        trie_sample = CorpusTrie(corpus=loader.corpus)
        # make sure load yields the same trie with list based construction
        assert len(trie_test) == len(trie_sample)
        # make sure the trie contain the same number of sentencees with corpus
        assert len(trie_test.make_list()) == len(loader.corpus)
        # path to large corpus
        PATH_CORPUS_MASTER = os.path.join('test', 'data', 'search_space.csv')
        # load corpus and show  prograress
        trie_test_large = CorpusTrie(corpus_path=PATH_CORPUS_MASTER,
                               hide_progress=False)
        loader = DataLoader(small_corpus=False)
        trie_sample_large = CorpusTrie(corpus=loader.corpus)
        # make sure load yields the same trie with list based construction
        assert len(trie_test_large) == len(trie_sample_large)
        # make sure the trie contain the same number of sentencees with corpus
        assert len(trie_test_large.make_list()) == len(trie_sample_large.make_list())
