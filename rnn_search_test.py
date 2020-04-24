import pytest

from zencorpora.rnn_search import SearchSpace, Hypothesis, HypothesesList
from zencorpora.corpustrie import CorpusTrie, TrieNode


class TestHypothesis:

    def test_overrides(self):
        """ Test overriden methods """
        # check if empty initializer works
        trie = CorpusTrie()
        hyp = Hypothesis(node=trie.root)
        assert hyp.node == trie.root
        # check repr
        assert float(repr(hyp)) == 0
        trie.insert(['this', 'is', 'test'])
        # check if default initializer works
        hyp2 = Hypothesis(node=trie.root.children[-1],
                          lprob=0.03,
                          parent_hyp=hyp)
        # lprob = 0 + -0.03
        assert hyp2.lprob == 0.03
        assert hyp2.parent_hyp == hyp
        # check comparison operator
        hyptemp = Hypothesis(node=TrieNode('it'),
                          lprob=0.02,
                          parent_hyp=hyp)
        # hyp3.lprob = 0.02 < hyp2.lprob = 0.03
        assert hyptemp < hyp2

    def test_chain(self):
        """ Test trace back method """
        trie = CorpusTrie()
        hyp = Hypothesis(node=trie.root)
        assert hyp.node == trie.root
        trie.insert(['this', 'is', 'test'])
        # check if default initializer works
        hyp2 = Hypothesis(node=trie.root.children[-1],
                          lprob=0.03,
                          parent_hyp=hyp)
        # check if hypothesis forms a chain going back to root
        hyp3 = Hypothesis(node=hyp2.node.get_child(-1),
                          lprob=0.03,
                          parent_hyp=hyp2)
        assert repr(hyp3.node) == 'is'
        hyp4 = Hypothesis(node=hyp3.node.get_child(-1),
                          lprob=0.03,
                          parent_hyp=hyp3)
        assert (hyp4.parent_hyp == hyp3) and \
               (hyp3.parent_hyp == hyp2) and \
               (hyp2.parent_hyp == hyp)
        assert hyp4.lprob == 0.09
        # check trace back recovers a sentence
        recovered_sentence = hyp4.trace_back()
        assert recovered_sentence == 'this is test'

    def test_traceback_in_loop(self):
        """ Test if trace back method works in a loop """
        trie = CorpusTrie()
        trie.insert(['this', 'is', 'test', 'code'])
        curr_hyp = Hypothesis(node=trie.root)
        hyp_list = [curr_hyp]
        while not curr_hyp.node.is_leaf():
            new_hyp = Hypothesis(node=curr_hyp.node.get_child(-1),
                                 lprob=0.1,
                                 parent_hyp=curr_hyp)
            hyp_list[-1] = new_hyp
            curr_hyp = new_hyp
        result = hyp_list[-1]
        recovered_sentence = result.trace_back()
        assert recovered_sentence == 'this is test code'
        assert round(result.lprob, 1) == 0.4


class TestHypothesesList:

    def test_init(self):
        """ Test init """
        list = HypothesesList(5)
        assert list.max_len == 5
        # check if an object inherits parent class correctly
        assert len(list) == 0

    def test_add(self):
        """ Test add method """
        list = HypothesesList(5)
        dummy = TrieNode('dummy')
        # check if it accepts a single element correctly
        for i in range(3):
            hyp = Hypothesis(node=dummy)
            hyp.lprob = i/10
            list.add(hyp)
        assert len(list) == 3
        # check clear fundtion in parent class works
        list.clear()
        assert len(list) == 0
        # check if it accepts a list
        hyps = []
        for i in range(5):
            hyp = Hypothesis(node=dummy)
            hyp.lprob = i/10
            hyps.append(hyp)
        list.add(hyps)
        assert len(list) == 5
        # check if the list is iterable
        identical = True
        for h, l in zip(hyps, list):
            if h != l:
                identical = False
        assert identical

    def test_capacity(self):
        """ Test if list maintains its maximum length """
        list = HypothesesList(5)
        dummy = TrieNode('dummy')
        # it shouldn't store more than 5 elements inside
        for i in range(100):
            hyp = Hypothesis(node=dummy)
            hyp.lprob = i
            list.add(hyp)
        assert len(list) == 5
        # list should store the 5 largest values, i.e. 95, ..., 99
        ordered = True
        for l, i in zip(list, range(95, 100)):
            if l.lprob != i:
                ordered = False
        assert ordered

    def test_isend(self):
        """ Test is end method (check searcher reaches the end of trie)"""
        list = HypothesesList(5)
        dummy = TrieNode('dummy')
        hyps = [Hypothesis(dummy) for _ in range(5)]
        list.add(hyps)
        # since dummy hypotheses are leaf nodes, method should return True
        assert list.is_end()
        dummy.children = [TrieNode('d')]
        list.add(Hypothesis(dummy))
        # since one node is not leaf, method shold return False
        assert not list.is_end()


class SearchSpace:

    def test_text2tensor(self):
        """ Test text2tensor method """
        pass
