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

    def remove(self, token):
        idx = self.find(token)
        if idx >= 0:
            self.children.pop(idx)


class CorpusTrie():

    def __init__(self, corpus=None, case_sensitive=False):
        self.root = TrieNode('<root>')
        self.num_token = 0
        self.case_sensitive = case_sensitive
        if corpus is not None:
            self.update(corpus)

    def __len__(self):
        return self.num_token

    def __contains__(self, sentence):
        if not self.case_sensitive:
            sentence = [token.lower() for token in sentence]
        curr_node = self.root
        for token in sentence:
            idx = curr_node.find(token)
            if idx == -1:
                return False
            curr_node = curr_node.get_child(idx)
        return True

    def insert(self, sentence):
        if not isinstance(sentence, list):
            raise AttributeError('input sentence should be tokenized.')
        if not self.case_sensitive:
            sentence = [token.lower() for token in sentence]
        curr_node = self.root
        # traverse tree until it finds path doesn't exist
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

    def remove(self, sentence):
        if not isinstance(sentence, list):
            raise AttributeError('input sentence should be tokenized.')
        if self.__len__ == 0:
            return -1
        if not self.case_sensitive:
            sentence = [token.lower() for token in sentence]
        curr_node = self.root
        sentence_path = [curr_node]
        for token in sentence:
            child_idx = curr_node.find(token)
            if child_idx >= 0:
                curr_node = curr_node.get_child(child_idx)
                sentence_path.append(curr_node)
            # if token doesn't exist, input sentence doesn't exits
            else:
                return -1
        # if no path matches input sentence, returns -1
        if not curr_node.is_leaf():
            return -1
        # finds the node to cut off from tree
        sentence_path.reverse()
        num_remove = 1
        # resulting node will be the root of path of nodes with only one child
        for idx, node in enumerate(sentence_path):
            try:
                parent = sentence_path[idx+1]
                if len(parent) == 1:
                    num_remove += 1
                else:
                    break
            except IndexError:
                break
        if node == '<root>':
            self.num_token = 0
            self.node.children.empty()
        else:
            sentence_path[idx+1].remove(node.token)
            self.num_token -= num_remove
        return 1

    def update(self, corpus):
        if not isinstance(corpus, list):
            raise AttributeError('corpus should be a list of sentences.')
        if (not isinstance(corpus[0], list)):
            raise AttributeError('input sentences should be tokenized.')
        for sentence in corpus:
            self.insert(sentence)

    def make_list(self):
        root = []
        self._make_list(root, [], self.root)
        return root

    def _make_list(self, root, path, curr_node):
        for node in curr_node.children:
            if node.is_leaf():
                sentence = path + [node.token]
                root.append(sentence)
            else:
                self._make_list(root, path+[node.token], node)

    def save_corpus(self, output_path):
        corpus = self.make_list()
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in corpus:
                sentence = ' '.join(sentence)
                f.write(sentence + '\n')
