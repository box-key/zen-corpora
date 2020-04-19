class TrieNode():


    def __init__(self, token):
        self.token = token
        self.children = []


    def __repr__(self):
        return self.token


    def __len__(self):
        return len(self.children)


    def is_leaf(self):
        return len(self.children) == 0


    def add_child(self, token):
        if self.find(token) == -1:
            new_node = TrieNode(token)
            self.children.append(new_node)


    def find(self, token):
        for idx, node in enumerate(self.children):
             if token == node.token:
                 return idx
        return -1


    def get_child(self, index):
        try:
            return self.children[index]
        except IndexError:
            return -1


class CorpusTrie():


    def __init__(self, sentence, model=None):
        self.model = model
        self.root = TrieNode('')
        self.num_token = 0


    def __len__(self):
        return self.num_token


    def __contains__(self, sentence):
        curr_node = self.root
        for token in sentence:
            idx = curr_node.find(token)
            if idx == -1:
                return False
            curr_node = curr_node.get_child(idx)
        return True


    def insert(self, sentence):
        curr_node = self.root
        for idx, token in enumerate(sentence):
            child_idx = curr_node.find(token)
            if child_idx >= 0:
                curr_node = curr_node.get_child(child_idx)
            else:
                for rest in sentence[idx:]:
                    if rest != '':
                        curr_node.add_child(rest)
                        curr_node = curr_node.children[-1]
                        self.num_token += 1
                break


    def remove(self, sentence, delete_all=None):
        if self.__len__ == 0:
            return -1
        curr_node = self.root
        candidate = TrieNode('')
        for token in sentence:
            child_idx = curr_node.find(token)
            if child_idx >= 0:
                curr_node = curr_node.get_child(child_idx)
                if (len(curr_node) > 1):
                    candidate = curr_node
            else:
                return -1
        if (curr_node.is_leaf()) and (len(candidate) != 1):
            candidate = curr_node
        assert (delete_all is None) and (not candidate.is_leaf()), \
            'The sentence you specified is accosiated with ' \
            + len(curr_node) + ' tokens.\n' \
            + 'If you like to remove these tokens, set `delete_all` to be True.'
        del candidate
        return 1


    def update(self, corpus):
        assert not isinstance(corpus, list), \
            'corpus should be a list of tokenized sentences.'
        assert not isinstance(corpus[0], list), \
            'each sentence should be tokenized.'
        for sentence in corpus:
            self.insert(sentence)


    # def save():
    #
    # @classmethod
    # def load():
    #
    # def make_list():
