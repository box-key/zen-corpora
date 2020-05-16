from setuptools import Extension, setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()


__version__ = "0.1.2"


setup(
    name="zen-corpora",
    version=__version__,
    author="Kei Nemoto",
    author_email="kei.nemoto28@gmail.com",
    description="corpus-level trie to store corpus efficiently and speed up sentence search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/box-key/zen-corpora",
    keywords=[
        "text search",
        "natural language understanding",
        "beam search",
        "recurrent neural network",
        "language modeling"
    ],
    install_requires=["sortedcontainers>=2.1", "tqdm>=4.31"],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    packages=find_packages(),
    license="Apache 2.0",
    zip_safe=False
)
