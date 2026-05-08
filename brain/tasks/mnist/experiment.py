"""MNIST experiment — does the substrate generalize across modalities?

Loads MNIST via sklearn, trains substrate-LM-style with supervised
reward, evaluates accuracy. Run:

  PYTHONPATH=. python3 tasks/mnist/experiment.py

By default uses 5000 train / 1000 test for fast iteration.
Override with TRAIN_N=60000 TEST_N=10000 for the full set.
"""

from __future__ import annotations

import os
import time

import numpy as np

from brain.tasks.mnist.mnist import build_mnist_brain
from brain.tasks.mnist.encoder import ImageEncoder
from brain.tasks.mnist.fast import (
    build_dense_view, sync_dense_to_brain,
    fast_train_epoch, fast_evaluate, verify_substrate_learning,
)


def load_mnist(*, train_n: int = 5000, test_n: int = 1000,
                seed: int = 0) -> dict:
    """Load MNIST via sklearn (downloads on first run, caches after).
    Returns dict with X_train, y_train, X_test, y_test as float arrays."""
    from sklearn.datasets import fetch_openml
    print(f'Loading MNIST (train_n={train_n}, test_n={test_n})...')
    mnist = fetch_openml('mnist_784', version=1, as_frame=False,
                          parser='liac-arff')
    X_all = mnist.data.astype(np.float32) / 255.0
    y_all = mnist.target.astype(np.int64)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_all))
    train_idx = idx[:train_n]
    test_idx = idx[train_n:train_n + test_n]

    X_train = X_all[train_idx].reshape(-1, 28, 28)
    y_train = y_all[train_idx]
    X_test = X_all[test_idx].reshape(-1, 28, 28)
    y_test = y_all[test_idx]

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
    }


def main():
    train_n = int(os.environ.get('TRAIN_N', '5000'))
    test_n = int(os.environ.get('TEST_N', '1000'))
    epochs = int(os.environ.get('EPOCHS', '5'))
    eta = float(os.environ.get('ETA', '0.01'))
    grid = int(os.environ.get('GRID', '16'))
    levels = int(os.environ.get('LEVELS', '4'))
    batch_size = int(os.environ.get('BATCH', '64'))
    optimizer = os.environ.get('OPT', 'adam')

    encoder = ImageEncoder(grid_size=grid, n_levels=levels)

    data = load_mnist(train_n=train_n, test_n=test_n)

    brain, vocab, encoder = build_mnist_brain(encoder=encoder)
    view = build_dense_view(brain, vocab, encoder)
    print(f'Encoder: {encoder.grid_size}×{encoder.grid_size} grid × '
          f'{encoder.n_levels} levels = {encoder.n_input_neurons()} input neurons')
    print(f'Brain: {brain.size} neurons, dense view {view.W.shape}')

    # Cold-start
    cold = fast_evaluate(view, data['X_test'], data['y_test'],
                          vocab=vocab, encoder=encoder)
    print(f'\nCold-start test accuracy: {cold["accuracy"]:.1%} '
          f'({cold["n_correct"]}/{cold["n_total"]}, {cold["n_blank"]} blank)')

    # Train
    print(f'\n=== Training ({epochs} epochs, eta={eta}, batch={batch_size}, opt={optimizer}) ===')
    rng = np.random.default_rng(0)
    best_test_acc = 0.0
    for ep in range(epochs):
        t0 = time.time()
        train_acc = fast_train_epoch(view,
                                       data['X_train'], data['y_train'],
                                       vocab=vocab, encoder=encoder,
                                       eta=eta, batch_size=batch_size,
                                       optimizer=optimizer,
                                       rng=rng)
        elapsed = time.time() - t0
        test_eval = fast_evaluate(view, data['X_test'], data['y_test'],
                                    vocab=vocab, encoder=encoder)
        if test_eval['accuracy'] > best_test_acc:
            best_test_acc = test_eval['accuracy']
        print(f'  ep {ep+1}/{epochs}  train_acc={train_acc:.1%}  '
              f'test_acc={test_eval["accuracy"]:.1%}  '
              f'(best {best_test_acc:.1%})  '
              f'time={elapsed:.1f}s')

    # Verify the SUBSTRATE itself learned — sync W → edges, then run
    # spread()-based predict on a sample and compare to fast_predict.
    print(f'\n=== Verifying substrate learning (spread() vs fast path) ===')
    verify = verify_substrate_learning(
        brain, vocab, encoder, view,
        data['X_test'], data['y_test'],
        n_samples=200,
    )
    print(f'  N samples: {verify["n_samples"]}')
    print(f'  fast_predict accuracy   : {verify["fast_accuracy"]:.1%}')
    print(f'  spread() accuracy       : {verify["substrate_accuracy"]:.1%}')
    print(f'  match rate (same preds) : {verify["fast_predict_matches_substrate"]:.1%}')
    print(f'  → substrate edges encode the learning (match rate near 1.0 confirms)')
    print(f'  → fast path is just a faster *layout* of the same edge weights')
    print(f'  Edges in substrate: {getattr(brain, "_used_synapses", 0)}')

    # Final evaluation
    final = fast_evaluate(view, data['X_test'], data['y_test'],
                            vocab=vocab, encoder=encoder)
    print(f'\n=== Final ===')
    print(f'  Test accuracy: {final["accuracy"]:.1%} '
          f'({final["n_correct"]}/{final["n_total"]})')
    print(f'  Brain: {brain.size} neurons, {getattr(brain, "_used_synapses", 0)} synapses')

    # Per-digit accuracy from confusion matrix
    confusion = final['confusion']
    print(f'\n  Per-digit accuracy:')
    for d in range(10):
        total_d = confusion[d].sum()
        correct_d = confusion[d, d]
        if total_d > 0:
            print(f'    digit {d}: {correct_d/total_d:.1%} '
                  f'({correct_d}/{total_d})')


if __name__ == '__main__':
    main()
