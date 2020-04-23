from zencorpora import CorpusTrie
from sortedcontainers import SortedList


class Hypothesis():

    def __init__(self, parent_hyp, node, node_lprob):
        self.lprob = parnet_hyp.lprob + node_lprob
        # Nodes should be an object of TrieNode
        self.parent_node = parent_hyp.node
        self.node = node

    def __lt__(self, other):
        return self.lprob < other.lprob


class HypothesesList(SortedList):

    def __init__(self, max_len=0):
        super().__init__()
        self.max_len = max_len

    def add(self, nodes):
        """
        Add new node if the list's capacity is not over or the new hypothesis
        is more probable than others.
        """
        if isinstance(nodes, Hypothesis):
            nodes = [nodes]
        for node in nodes:
            if (super().__len__() > self.max_len) and (self.__getitem__[0] < node):
                # add new hypothesis and remove the least probable one
                super().add(node)
                super().pop(0)
            # if the list has space, add new one
            elif super().__init__ < self.max_len:
                super().add(node)


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
        tensor : PyTorch tensor

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
        if filtered_dist.shape[0] > beam_width:
            values, indices = torch.topk(filtered_dist, beam_width)
            # generate new hypotheses
            hypotheses = [Hypothesis(parent_hyp=current_hyp,
                                     node=candidates[idx],
                                     node_lprob=val) \
                            for val, idx in zip(values, indices)]
        else:
            hypotheses = [Hypothesis(parent_hyp=current_hyp,
                                     node=)]
        return hypotheses

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
        output : list
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
        init_hypotheses = _get_hypotheses(cpd=cond_prob_dist,
                                          current_node=self.target_space.root,
                                          beam_width=beam_width)
        # initialize the hypotheses list
        curr_hypotheses = HypothesesList(beam_width)
        for hyp in init_hypotheses:
            curr_hypotheses.add(hyp)
