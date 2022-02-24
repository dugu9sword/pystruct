# Notes

## Overview

This note illustrates the basic design of this repository.

There are mainly three components:
- **Model**: A model only defines the topology of a graph. 
  - It defines how to convert a variable sized (x,) into xnary potentials (for SSVM, it may define loss-augmented xnary potentials for decoding), mainly for inference
  - It defines how to convert a variable sized (x, y) into a fix-sized vectors, mainly for parameter estimation 
  - The model itself does not define anything related to learning or inference

- **Inference**: Perform MAP inference based on xnary-potentials (from a model)
- **Learner**: Tuning the model's parameters

## Model

Take `GraphCRF` as an example. (`StructureModel` (base) -> `CRF` -> `GraphCRF`, `ChainCRF`, etc.)

**weight** is a **fix-sized** vector whose shape is `(n_states x n_features + n_states x nstates)`.

- *unary_param*: First `n_states x n_features` dims
- *binary_param*: Last `n_states x n_states` dims

**x** is a tuple (features, edges), different instances has different sizes, this can be viewed as **variable-sized** input!

- *features*: `n_nodes x n_features`, dotting `unary_param` to get *unary_potentials*
- *edges*: `n_edges x 2`, indexing `binary_param` to get *binary_potentials*
- Note that xary-potentials are also of variable size

**joint_feature** is a **fix-sized** vector whose shape is same to **weight**. 

- Is is feature vector of **(x, y)** instead of **x** 
- It is used mainly for training
- Note that (x, y)'s shape is different case-by-case since they have different number of nodes and edges
- We try to convert it into a fix-sized dense vector so that it can be directly multiplied with the weight vector

See the comments in `GraphCRF.joint_feature` of `pystruct/models/graph_crf.py` for details

## Learning

### Loss Augmented Decoding

See the comments in `CRF.loss_augmented_inference` of `pystruct/models/crf.py` for details

### Subgradient SSVM

#### Structured Perceptron
`pystruct/models/structured_perceptron.py` is the margin-free version of `pystruct/models/subgrdient_ssvm.py`. More specifically, a difference is whether `loss_augmented_inference` is used. 
