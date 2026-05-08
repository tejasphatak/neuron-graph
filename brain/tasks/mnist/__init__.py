"""MNIST classification on the substrate — modality polymorphism test.

Same primitives as TTT (RL games) and LM (sequence modeling), different
problem shape: static image → digit class. No sequence, no rollouts.
Just (active pixels → label) edges grown from supervised reward.

If the substrate works here without architectural change, modality
polymorphism is established.
"""

from .mnist import (
    MnistVocab, build_mnist_brain,
    encode_image, predict, train_step, train_epoch,
    evaluate,
)
from .encoder import ImageEncoder, ImageEncoderVocab

__all__ = [
    'MnistVocab', 'build_mnist_brain',
    'encode_image', 'predict', 'train_step', 'train_epoch',
    'evaluate',
    'ImageEncoder', 'ImageEncoderVocab',
]
