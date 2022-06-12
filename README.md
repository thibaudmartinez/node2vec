<p align="center">
  <h1 align="center">node2vec</h1>
  <p align="center">
    An efficient node2vec implementation in C++ with a Python API.
    <br>
     <a href="https://mybinder.org/v2/gh/thibaudmartinez/node2vec/master?filepath=demo.ipynb">View demo</a>
     ·
     <a href="https://github.com/thibaudmartinez/node2vec/issues">Report bug</a>
  </p>
</p>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/thibaudmartinez/node2vec/master?filepath=demo.ipynb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Build Status](https://travis-ci.com/thibaudmartinez/node2vec.svg?branch=master)](https://travis-ci.com/thibaudmartinez/node2vec)

## Table of contents

* [About](#about)
  * [Presentation](#presentation)
  * [Motivation](#motivation)
* [Getting started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
  * [Basic usage](#basic-usage)
  * [Instantiating the model from alternative graph formats](#instantiating-the-model-from-alternative-graph-formats)
  * [Dealing with directed and weighted edges](#dealing-with-directed-and-weighted-edges)
  * [Deterministic runs](#deterministic-runs)
* [API reference](#api-reference)
* [Building from source](#building-from-source)
* [Reporting a bug](#reporting-a-bug)
* [Acknowledgments](#acknowledgments)
* [References](#references)
* [License](#license)

## About

### Presentation

node2vec *[(Grover and Leskovec, 2016)](https://arxiv.org/abs/1607.00653)* is a machine
learning method used to create vector representations of the nodes of a graph. More 
information about node2vec can be found [here](https://snap.stanford.edu/node2vec/). 
This repository provides an efficient and convenient implementation of node2vec.
The implementation consists of a backend written in C++ linked to a Python API.

### Motivation

At the time of writing, there are two main implementations of node2vec available on 
the Internet: a reference [Python implementation](https://github.com/aditya-grover/node2vec)
and a high performance C++ implementation included in 
[Stanford Network Analysis Project (SNAP)](http://snap.stanford.edu/index.html).
This project aims at bringing together the best of the two implementations.
It uses node2vec SNAP implementation as a high performance backend to simulate the 
random walks on the graph. [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
is used to compute the node embeddings. Finally, the library can be manipulated through
an easy-to-use Python API.

## Getting started

### Prerequisites

* Linux 64 bits
* [Python 3.7, 3.8 or 3.9](https://www.python.org/downloads/)
* [GCC >= 8](https://gcc.gnu.org/gcc-8/changes.html) 

### Installation

Python 3.8
````bash
pip install https://github.com/thibaudmartinez/node2vec/releases/download/v0.3.0/node2vec-0.1.0-cp38-cp38-linux_x86_64.whl
````

Python 3.9
````bash
pip install https://github.com/thibaudmartinez/node2vec/releases/download/v0.3.0/node2vec-0.1.0-cp39-cp39-linux_x86_64.whl
````

## Usage

### Basic usage

The model expects as input a graph in which the nodes are identified by consecutive
integers starting from 0. The edgelist of the graph is used to instantiate the model.

````python
import numpy as np
from node2vec.model import Node2Vec

# Create an edgelist
src_nodes = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8], dtype=np.int32)
dest_nodes = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0], dtype=np.int32)

# Create a node2vec model from the edgelist
node2vec_model = Node2Vec(src_nodes, dest_nodes, graph_is_directed=False)

# Simulate biased random walks on the graph
node2vec_model.simulate_walks(
    walk_length=25,
    n_walks=10,
    p=1,
    q=0.5,
    workers=2
)

# Print the generated walks
print(node2vec_model.walks)

# Learn node embeddings from the generated random walks
node2vec_model.learn_embeddings(
    dimensions=64,
    context_size=10,
    epochs=2,
    workers=2
)

# Print the embedding corresponding to the first node
print(node2vec_model.embeddings[0])
````

### Instantiating the model from alternative graph formats

For convenience, the library support the creation of node2vec models from NetworkX
graphs and from adjacency matrices. 

**Note:** if you are manipulating large graphs or if you are after performance, the
recommended way to instantiate the model is to use an edgelist.

#### From a NetworkX graph

A node2vec model can be instantiated from a NetworkX
[Graph](https://networkx.github.io/documentation/stable/reference/classes/graph.html)
or [DiGraph](https://networkx.github.io/documentation/stable/reference/classes/digraph.html).

````python
import networkx as nx
from node2vec.model import Node2Vec

graph = nx.karate_club_graph()
node2vec_model = Node2Vec.from_nx_graph(graph)
````

#### From an adjacency matrix

It is also possible to create a model from an adjacency matrix stored as a numpy array.

````python
import numpy as np
from node2vec.model import Node2Vec

adj_matrix = np.array([
    [0, 1, 2, 0],
    [0, 0, 1, 2],
    [1, 0, 0, 1],
    [1, 0, 0, 0]
], dtype=np.double)

node2vec_model = Node2Vec.from_adj_matrix(adj_matrix)
````

### Dealing with directed and weighted edges

This node2vec implementation supports directed and undirected graphs, as well as 
weighted and unweighted graphs.

#### Directed and undirected graphs

* When using the class constructor, pass a boolean to the parameter `graph_is_directed`
to indicates whether the input edgelist corresponds to a directed or undirected graph.
Note that you don't need to symmetrize the edges in the list when dealing with an
undirected graph.

* When using the `from_nx_graph` method, directionality of the graph is automatically
taken into account depending on the type of NetworkX graph you input ([Graph](https://networkx.github.io/documentation/stable/reference/classes/graph.html)
or [DiGraph](https://networkx.github.io/documentation/stable/reference/classes/digraph.html)).

* When using the `from_adj_matrix` method, simply input the corresponding adjacency 
matrix: a symmetric matrix for an undirected graph and a non-symmetric one otherwise.  

#### Weighted and unweighted graphs

* When using the class constructor, pass a numpy array containing the edge weights if
you are dealing with a weighted graph, otherwise pass `None` to the `edge_weights`
parameter.

````python
src_nodes = np.array([1, 2, 2, 3, 3], dtype=np.int32)
dest_nodes = np.array([0, 0, 1, 0, 1], dtype=np.int32)

# Unweighted graph
model_u = Node2Vec(src_nodes, dest_nodes, edge_weights=None)

# Weighted graph
edge_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.double)
model_w = Node2Vec(src_nodes, dest_nodes, edge_weights=edge_weights)
````

* When using the `from_nx_graph` method, input a
[weighted NetworkX graph](https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_weighted_graph.html)
if you want to run node2vec on a weighted graph. That is, the NetworkX graph should have
a weight attribute on each edge.

* When using the `from_adj_matrix` method, simply indicate the weights of the edges in
the matrix, or put 1's in the matrix in the case of an unweighted graph.

### Deterministic runs

It is often necessary to ensure that a model will output the same results when run with
the same data and parameters. For instance, this is key in a scientific framework where
reproducibility of the results is essential.

To produce deterministic runs with node2vec, you must set the random seeds that will be
used at each step of the process. In addition, you must use a single worker thread, as
OS thread scheduling induce ordering jitter. Finally, you must set the PYTHONHASHSEED
environment variable to control hash randomization.

````python
import numpy as np
import os
from node2vec.model import Node2Vec

src_nodes = np.array([1, 2, 2, 3, 3], dtype=np.int32)
dest_nodes = np.array([0, 0, 1, 0, 1], dtype=np.int32)

SEED = 19
os.environ["PYTHONHASHSEED"] = str(SEED)

node2vec_model = Node2Vec(src_nodes, dest_nodes)
node2vec_model.simulate_walks(workers=1, rand_seed=SEED)
node2vec_model.learn_embeddings(workers=1, rand_seed=SEED)
````

## API reference

***class* node2vec.model.Node2Vec**(src_nodes, dest_nodes, edge_weights=None, graph_is_directed=False)

___
*Attributes*
___

**walks**: ndarray[np.int32]
<br>&nbsp;&nbsp;&nbsp;&nbsp;The biased random walks generated when running the simulate_walks method.

**embeddings**: ndarray[np.int32]
<br>&nbsp;&nbsp;&nbsp;&nbsp;The node embeddings generated by the Skip-Gram model from the biased random
walks.

**word2vec_model**: [gensim.models.Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
<br>&nbsp;&nbsp;&nbsp;&nbsp;The Gensim Word2Vec model used to generate the vector embeddings.

___
*Parameters*
___

**src_nodes**: ndarray[np.int32]
<br>&nbsp;&nbsp;&nbsp;&nbsp;Identifiers of the source nodes for each edge.
The length of the array is equal to the number of edges in the graph.

**dest_nodes**: ndarray[np.int32]
<br>&nbsp;&nbsp;&nbsp;&nbsp;Identifiers of the destination nodes for each edge.
The length of the array is equal to the number of edges in the graph.

**edge_weights**: ndarray[np.double] *(default: None)*
<br>&nbsp;&nbsp;&nbsp;&nbsp;Weights of the edges.
If the graph is unweighted leave empty or pass None.

**graph_is_directed**: bool *(default: False)*
<br>&nbsp;&nbsp;&nbsp;&nbsp;Indicates whether the graph is directed.
Default behaviour is to consider that the graph is undirected. 

___
*Methods*
___

**from_nx_graph**(nx_graph)
<br>&nbsp;&nbsp;&nbsp;&nbsp;Instantiate a node2vec model from a NetworkX graph.

&nbsp;&nbsp;&nbsp;&nbsp;*Note: when dealing with large graphs prefer to use the class constructor for
more efficiency.*

| Parameters |
|:-----------|
|**nx_graph**: [networkx.Graph](https://networkx.github.io/documentation/stable/reference/classes/graph.html) or [networkx.DiGraph](https://networkx.github.io/documentation/stable/reference/classes/digraph.html)<br>The graph (directed or not) that will be used by node2vec. |


|   Returns  |
|:-----------|
| An instantiated **node2vec.model.Node2Vec**. |

**from_adj_matrix**(adj_matrix)
<br>&nbsp;&nbsp;&nbsp;&nbsp;Instantiate a node2vec model from an adjacency matrix inputted as a dense
numpy array.
<br>&nbsp;&nbsp;&nbsp;&nbsp;The adjacency matrix can correspond to a directed or an undirected
graph.

&nbsp;&nbsp;&nbsp;&nbsp;*Note: when dealing with large graphs prefer to use the class constructor for
more efficiency.*

| Parameters |
|:-----------|
|**adj_matrix**: ndarray[np.double] of size n_nodes * n_nodes<br>The adjacency matrix of the graph that will be used by node2vec. |

|   Returns  |
|:-----------|
| An instantiated **node2vec.model.Node2Vec**. |

**simulate_walks**(walk_length=80, n_walks=10, p=1.0, q=1.0, workers=USABLE_CPUS, verbose=True, rand_seed=None)
<br>&nbsp;&nbsp;&nbsp;&nbsp;Repeatedly simulate random walks from each node.

| Parameters |
|:-----------|
|**walk_length**: int *(default: 80)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Length of walk per source node. |
|**n_walks**: int *(default: 10)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Number of walks per source node. |
|**p**: int *(default: 1)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Return hyperparameter. |
|**q**: int *(default: 1)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Inout hyperparameter. |
|**workers**: *(default: USABLE_CPUS)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Number of worker threads used. |
|**verbose**: bool *(default: True)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Verbosity of the output. |
|**rand_seed**: int *(default: None)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Seed for the random number generator.<br>&nbsp;&nbsp;&nbsp;&nbsp;Use a fixed random seed to produce deterministic runs.<br>&nbsp;&nbsp;&nbsp;&nbsp;To ensure fully deterministic runs you also need to set workers = 1. |

**learn_embeddings**(dimensions=128, context_size=10, epochs=1, workers=USABLE_CPUS, verbose=True, rand_seed=None)
<br>&nbsp;&nbsp;&nbsp;&nbsp;Learn embeddings using the Skip-Gram model.


| Parameters |
|:-----------|
| **dimensions**: int *(default: 128)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Number of dimensions of the generated vector embeddings. |
| **context_size**: int *(default: 10)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Context size (window size) in Word2Vec. |
| **epochs**: int *(default: 1)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Number of epochs in stochastic gradient descent. |
| **workers**: *(default: USABLE_CPUS)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Number of worker threads used. |
| **verbose**: bool *(default: True)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Verbosity of the output. |
| **rand_seed**: int *(default: True)*<br>&nbsp;&nbsp;&nbsp;&nbsp;Seed for the random number generator.<br>&nbsp;&nbsp;&nbsp;&nbsp;Use a fixed random seed to produce deterministic runs.<br>&nbsp;&nbsp;&nbsp;&nbsp;To ensure fully deterministic runs you also need to set workers = 1.<br>&nbsp;&nbsp;&nbsp;&nbsp;Reproducibility between Python interpreter launches also requires use of the PYTHONHASHSEED environment variable to control hash randomization. |

## Building from source

Clone the repository.
````bash
git clone https://github.com/thibaudmartinez/node2vec.git
````

Install [pybind11](https://github.com/pybind/pybind11).
````bash
pip install $(grep pybind11 requirements.txt)
````

Build and install node2vec.
````
pip install ./node2vec
````

## Reporting a bug

If you encounter a bug, please [fill an issue](https://github.com/thibaudmartinez/node2vec/issues).

## Acknowledgments

Many thanks to those behind these awesome libraries that are used extensively in
this project:

* [SNAP](http://snap.stanford.edu/)
* [Gensim](https://radimrehurek.com/gensim/)
* [pybind11](https://github.com/pybind/pybind11)

## References

Grover, A. and Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks.
In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery
and Data Mining - KDD ’16*, pages 855–864, San Francisco, California, USA. ACM Press.

## License

This implementation of node2vec is, for its original parts, licensed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
Copyright (c) 2019, Thibaud Martinez. All rights reserved.

The code derived from [SNAP](http://snap.stanford.edu/) is licensed
under its own terms, accessible [here](https://raw.githubusercontent.com/snap-stanford/snap/master/License.txt).
Copyright (c) 2007-2019, Jure Leskovec. All rights reserved.
