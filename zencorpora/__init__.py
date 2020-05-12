from .corpustrie import CorpusTrie
from .rnn_search import SearchSpace

import pkg_resources


# __version__ = pkg_resources.get_distribution("zen-corpora").version
__version__ = "0.1.1"

__all__ = [
    "CorpusTrie",
    "SearchSpace"
]
