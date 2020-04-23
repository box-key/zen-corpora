from zencorpora import CorpusTrie
from sortedcontainers import SortedList


class Hypothesis():
    """
    Store an object of TrieNode and its parent.
    This class also stores the log conditional probability of a sequence ends
    at the input node.
    """
    def __init__(self, parent_hyp=None, node, node_lprob):
        if parent_hyp is None:
            self.lprob = 0
            self.parent_node = None
        else:
            self.lprob = parnet_hyp.lprob + node_lprob
            # Nodes should be an object of TrieNode
            self.parent_node = parent_hyp.node
        self.node = node

    def __lt__(self, other):
        return self.lprob < other.lprob


class HypothesesList(SortedList):
    """
    This class is used to maintain hypothesis given by beam search, where all
    elements are stored in ascending order.
    It only stores maximum length (beam width) of elements. If a new element is
    smaller than any elements in the list, it discards the new element.
    """
    def __init__(self, max_len=0):
        super().__init__()
        self.max_len = max_len

    def add(self, hypotheses):
        """
        Add new node if the list's capacity is not over or the new hypothesis
        is more probable than others.
        """
        if isinstance(hypotheses, Hypothesis):
            hypotheses = [hypotheses]
        for hyp in hypotheses:
            if (super().__len__() > self.max_len) and (self.__getitem__[0] < hyp):
                # add new hypothesis and remove the least probable one
                super().add(hyp)
                super().pop(0)
            # if the list has space, add new one
            elif super().__init__ < self.max_len:
                super().add(hyp)

    def is_end(self):
        for hyp in super()._list:
            if not hyp.node.is_leaf():
                return False
        return True

class SearchSpace():
    """ A place to store corpus-trie and performs beam search.

    Attributes
    ----------
    src_field : Torchtext.Field
        An unique mapper between tokens and their id for source.
    trg_field : Torchtext.Field
        An unique mapper between tokens and their id for target.
    encoder : PyTorch Model
        Encodes input text into a hidden vector and pass it to decoder.
    decoder : PyTorch Model
        Generates conditional probability of current sequence given encoded
        input. It also generates hidden states in a recurrent manner.
    prob_generator : func
        Computes the negative log probability distribution of a sequence.
    target_corpus : list
        A list of sentences which are the search objectives of a model
    short_length_penalty : int
        Penalizes short sentences since the outcome is the sum of log negative
        probability.
    case_sensitive : bool
        Used to construct corpus trie. True by default.

    """

    def __init__(self,
                 src_field,
                 trg_field,
                 encoder,
                 decoder,
                 target_corpus,
                 score_function,
                 device,
                 short_length_penalty=1,
                 case_sensitive=True):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required, please run `pip install pytorch`")
        self.src_field= src_field
        self.trg_field = trg_field
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.score_function = score_function
        self.target_space = CorpusTrie(target_corpus, case_sensitive)

    def _text2tensor(self, sentence, src):
        """
        Parameters
        ----------
        sentence : list
            A tokenized sentence.
        src : bool
            To tell which field to be used.

        Return
        -------
        tensor : tensor
            Input mapped into target vocabulary space.

        """
        if not self.case_sensitive:
            sentence = [token.lower() for token in sentence]
        if src:
            sentence = [self.src_field.init_token] + \
                       sentence + [self.src_field.eos_token]
            mapped = [self.src_field.vocab.stoi[token] for token in sentence]
        else:
            sentence = sentence + [self.trg_field.eos_token]
            mapped = [self.trg_field.vocab.stoi[token] for token in sentence]
        tensor = torch.LongTensor(mapped).to(self.deivce)
        # tensor = [sentence_len]
        tensor = tensor.unsqueeze(1)
        # tensor = [sentence_len, 1]
        return tensor

    def _get_hypotheses(self, cpd, current_hyp, beam_width):
        """
        Parameters
        ----------
        cpd : tensor[trg_vocab]
            The conditional probability distribution over the target vocabulary.
        current_hyp : Hypothesis node
            A node at the current level.

        Returns
        -------
        hypotheses : list of Hypothesis
            A list of hypotheses based on the generated conditional probability
            distribution. Each node contains the conditional log probability of
            the hypothesis (a sequence of tokens which ends at the node).

        """
        candidates = current_hyp.node.children
        # map tokens into id in the vocabulary
        token_id = [self.trg_field.vocab.stoi[node.token] \
                        for node in candidates]
        # only retain tokens appear in candidates
        filtered_dist = cpd[token_id]
        # get the top element within beam_width
        # avoid index out of range error in torch.topk
        k = beam_width if filtered_dist.shape[0] else filtered_dist.shape[0]
        values, indices = torch.topk(filtered_dist, k)
        # generate new hypotheses
        hypotheses = [Hypothesis(parent_hyp=current_hyp,
                                 node=candidates[idx],
                                 node_lprob=val) \
                     for val, idx in zip(values.tolist(), indices.tolist())]
        return hypotheses

    def _hyp2text(hypotheses):
        """ It converts an object of hypoth class into a sentence.
        Parameters
        ----------
        hypotheses : a list of hypothesis
            A hypothesis given by beam serach.

        Returns
        -------
        sentences : a list of str
            A list of sentences given by hypotheses.
        """
        pass

    def beam_search(self, src, beam_width):
        """
        Parameters
        ----------
        src : list
            A tokenized sentence.
        beam : int
            The beam width for beam search.

        Returns
        -------
        result : list
            A list of sentences found by beam search
        """
        if isinstance(src, list):
            raise AttributeError('Input sentence should be tokenized.')
        if not self.case_sensitive:
            src = [token.lower() for token in src]
        # map input text into tenser
        src_tensor = _text2tensor(src)
        src_len = torch.tensor([src_tensor.shape[0]]).to(self.device)
        # encode input text
        enc_output = self.encoder(src_tensor, src_len)
        # get the first estimation given '<sos>' token
        sos_token = self.trg_field.vocab.stoi['<sos>']
        hypothesis = HypothesesList(beam_width)
        cond_prob_dist, hidden = self.decoder(sos_token, enc_output)
        # get beam_width number of hypothesis under the root
        init_hyp = Hypothesis(self.target_space.root, 0)
        init_hypotheses = _get_hypotheses(cpd=cond_prob_dist,
                                          current_node=init_hyp,
                                          beam_width=beam_width)
        # initialize the hypotheses list
        curr_hypotheses = HypothesesList(beam_width)
        for hyp in init_hypotheses:
            curr_hypotheses.add(hyp)
        while not curr_hypotheses.is_end():
            next_hypotheses = HypothesesList(beam_width)
            for hyp in curr_hypotheses:
                dec_input = self.trg_field.vocab.stoi[hyp.node.token]
                cond_prob_dist, hidden = self.decoder(input, hidden)
                # update current list
                next_hypotheses.add(
                    _get_hypotheses(cpd=cond_prob_dist,
                                    current_node=hyp,
                                    beam_width=beam_width)
                )
            # swao two lists
            curr_hypotheses = next_hypotheses
        result = _hyp2text(curr_hypotheses)
        return result
