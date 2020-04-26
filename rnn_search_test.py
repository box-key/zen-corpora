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
        for h, l in zip(hyps, list):
            assert h == l

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
        for l, i in zip(list, range(95, 100)):
            assert l.lprob == i

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


from torch.nn.functional import log_softmax

# Initialize SearchSpace and models
from test.loader import DataLoader
data = DataLoader()
space = SearchSpace(
    src_field = data.input_field,
    trg_field = data.output_field,
    encoder = data.model.encoder,
    decoder = data.model.decoder,
    target_corpus = data.corpus,
    score_function = log_softmax,
    device = data.device,
)


class TestSearchSpace:

    def test_target_space(self):
        """ Make sure initializer constructs target space from input corpus """
        target_corpus = space.target_space.make_list()
        assert len(target_corpus) == 10
        for t, d in zip(target_corpus, data.corpus):
            assert t == d

    def test_text2tensor(self):
        """ Test text2tensor method """
        test = ['this', 'is', 'a', 'test']
        # check method for src argument
        test_num = [space.src_field.vocab.stoi[token] for token in test]
        sos = space.src_field.vocab.stoi['<sos>']
        eos = space.src_field.vocab.stoi['<eos>']
        test_num = [sos] + test_num + [eos]
        tensor = space._text2tensor(test, src=True)
        # check shape of output tensor
        # the shape should be [<eos> + 4 input tookens + <eos>, 1]
        assert tensor.shape[0] == 6
        assert tensor.shape[1] == 1
        # check each output
        for t, o in zip(tensor, test_num):
            assert t == o
        # check method for trg argument
        test_num = [space.trg_field.vocab.stoi[token] for token in test]
        sos = space.trg_field.vocab.stoi['<sos>']
        eos = space.trg_field.vocab.stoi['<eos>']
        test_num = test_num + [eos]
        tensor = space._text2tensor(test, src=False)
        # check shape of output tensor
        # the shape should be [<eos> + 4 input tookens + <eos>, 1]
        assert tensor.shape[0] == 5
        assert tensor.shape[1] == 1
        # check each output
        identical = True
        for t, o in zip(tensor, test_num):
            if t != o:
                identical = False
        assert identical

    def test_hyp2text(self):
        """ Test hyp2text method """
        text1 = ['this', 'is', 'test', 'code']
        text2 = ['this', 'is', 'test']
        text3 = ['it', 'aint']
        texts = [' '.join(text1), ' '.join(text2), ' '.join(text3)]
        hyp1 = self.text2hyp(text1)
        hyp2 = self.text2hyp(text2)
        hyp3 = self.text2hyp(text3)
        # just make sure text2hyp works
        assert round(float(repr(hyp1)), 1) == 0.4
        assert round(float(repr(hyp2)), 1) == 0.3
        hyps = [hyp1, hyp2, hyp3]
        outs = space._hyp2text(hyps)
        # make sure hyp2text returns the same number of sentences
        assert len(outs) == 3
        # make sure hyp2text recovers exactly the same sentences
        identical = True
        for o, t in zip(outs, texts):
            if o != t:
                identical = False
        assert identical

    def text2hyp(self, text):
        curr_hyp = Hypothesis(node=TrieNode('<root>'))
        for token in text:
            next_hyp = Hypothesis(node=TrieNode(token),
                                  lprob=0.1,
                                  parent_hyp=curr_hyp)
            curr_hyp = next_hyp
        return curr_hyp


    def test_extract_top_hypotheses(self):
        """ Test get hypotheses method """
        import torch
        src = ['this', 'is', 'test']
        # make encoder inputs
        src_tensor = space._text2tensor(src, src=True)
        src_len = torch.tensor([src_tensor.shape[0]])
        # encode inputs
        enc_output = space.encoder(src_tensor, src_len)
        # initial decoding with <sos> token given src
        sos_token = space.trg_field.vocab.stoi['<sos>']
        sos_token = torch.tensor([sos_token], dtype=torch.long)
        # generate prob dist over target vocabulary
        cond_prob_dist, hidden = space.decoder(sos_token, enc_output)
        init_hyp = Hypothesis(node=space.target_space.root)
        # check _get_hypotheses method only returns probability of candidates
        init_hypotheses = space._get_hypotheses(cpd=cond_prob_dist,
                                                current_hyp=init_hyp)
        assert len(init_hypotheses) == 9
        # if beam width exceeds next candidates
        # make sure hypotheses list retains beam width number of hypotheses
        list = HypothesesList(4)
        list.add(init_hypotheses)
        assert len(list) == 4
        # check if list retains the top elements among candidates
        candidates = space.target_space.root.children
        token_id = [space.trg_field.vocab.stoi[node.token] \
                        for node in candidates]
        cpd = cond_prob_dist.squeeze(0)
        val, idx = torch.topk(cpd[token_id], 4)
        val = val.tolist()
        # make the topk list ascending order
        val.reverse()
        for v, l in zip(val, list):
            assert round(v, 3) == round(float(repr(l)), 3)

    def test_beam_search(self):
        """ Test beam search method """
        src = ['this', 'is', 'test']
        # check if it returns beam width number of sentences
        result = space.beam_search(src, 2)
        assert len(result) == 2
        # check again
        result = space.beam_search(src, 4)
        assert len(result) == 4
        # check if it returns the maximum length if beam width exceeds trie size
        result = space.beam_search(src, 1000)
        assert len(result) == 10

src = ['I', 'want', 'horror', 'movie']
# check if it returns beam width number of sentences
result = space.beam_search(src, 3)
print(result)
