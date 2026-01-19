# deep-learning-foundations
Implementation of core Deep Learning architectures from scratch.

This repository is a from-scratch exploration of core neural network architectures, implemented with the explicit goal of understanding why they work.
The emphasis was on:
* Understanding architectures from first principles
* Observing how design choices affect learning
* Building intuition

## What I learned

1. Bigram Language Model `Bigram!.ipynb`
   * Implemented a count-based statistical baseline that directly estimates next-character probabilities and achieves an average negative log-likelihood of ~2.45.
   * Reimplemented the same model as a single-layer neural network trained with gradient descent.
   * The neural model moves to a similar loss (~2.49), showing it learns the same bigram statistics in a parameterized form.
   * Learned that neural networks are essentially probability estimators, and that the performance limit comes from the bigram assumption, not training or optimization.

2. MLP with Batch Normalization `MLP_in_2_Ways.ipynb`
   * Built a character-level language model using a deep MLP with a fixed context window instead of bigrams.
   * Started with a fully manual implementation, writing every operation (embedding, linear layers, batch norm, activation) explicitly to understand what actually happens during forward and backward passes.
   * Implemented Kaiming initialization to prevent vanishing gradients.
   * Implemented Batch Normalization from scratch, including running mean and variance.
   * Training converges to a much lower loss (~2.0 on train/val) compared to the bigram model.
   * Refactored the model into a class architecture (Linear, BatchNorm1d, Tanh) trying to mimic `torch.nn`.
   * Implemented training vs inference mode handling.

3. Wavenet `wavenet.ipynb`
   * Built a character-level language model with an 8-character context that progressively combines nearby characters using a hierarchical structure based on grouped convolutions (FlattenConsecutive).
   * Wrote all core layers from scratch (Embedding, Linear, BatchNorm1d, Tanh) instead of using torch.nn.
   * Introduced a FlattenConsecutive layer that groups characters together, increasing the model’s effective context depth by depth.
   * Implemented a Sequential class to run layers in order, manage parameters, and switch between training and generation modes
   * Achieves lower loss (~1.9 train, ~2.0 validation/test) compared to earlier MLP models.

4. Decoder-only Transformer `decoder_only_transformer.ipynb`
   * Built a decoder-only Transformer for character level text generation, trained on the TinyShakespeare dataset.
   * Used a fixed context window (block_size = 256) and trained the model to predict the next character.
   * Implemented causal self-attention from scratch using masking so tokens cannot see future positions.
   * Implemented multi-head attention by running several attention heads in parallel and concatenating their outputs.
   * Used residual connections and Layer Normalization inside each Transformer block ```python
x = x + sublayer(LN(x))
to stabilize deep training.
   * Stacked multiple Transformer blocks (n_layer = 6) to increase model depth.
   * Trained the model using cross-entropy loss, reaching around ~1.1 train loss and ~1.5 validation loss.


## Acknowledgments

This repository follows the ideas and progression of Andrej Karpathy’s *Neural Networks: Zero to Hero* series.  
