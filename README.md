HeuGAT: A Hybrid Heuristic-Enhanced Graph Attention Framework
1. Introduction
HeuGAT is a novel multi-layered hybrid framework designed to address two fundamental tasks in social network analysis: Link Prediction and Breakup (Rift) Prediction. Building upon the architectural foundation of ClasReg, HeuGAT integrates deep learning (specifically, Graph Attention Networks) with classical heuristics to enhance predictive accuracy and interpretability in both sparse and dense graph settings.

The framework is composed of five key layers:

A Preprocessing Layer for transforming raw network data into model-ready format;

A Graph Attention Network (GAT) Embedding Layer for learning node representations through attention-based neighborhood aggregation;

A Classification Layer to predict whether a tie between two nodes exists;

A Regression Layer to estimate the strength or likelihood of an existing tie dissolving;

A Heuristic Layer that combines classification and regression results for final breakup inference.

HeuGAT retains the bifunctional capability of its predecessor, ClasReg, but introduces attention mechanisms to better capture structural dynamics in complex and heterogeneous social graphs.

2. Repository Structure and Dependencies
HeuGAT builds on the modular architecture of ClasReg. For implementation details, experimental setup, and class dependencies, users are encouraged to consult the original ClasReg GitHub repository. HeuGAT introduces modifications to the representation learning layer, replacing the original embedding strategy with Graph Attention Networks, and adjusts the inference pipeline accordingly.
