"""Linear-chain Conditional Random Field (Lafferty 2001) re-implemented
as a ``paddle.nn.Layer``.

The CRF sits on top of an emission-score module (here a single
``nn.Linear`` for convenience, but any ``[T, n_tags]`` tensor can
be passed in). The transition matrix is a learnable parameter, and
the forward pass returns per-position marginal posteriors plus the
log partition function — both differentiable, so the whole thing
trains with standard backprop.
"""

import paddle

from typing import Tuple


def _logsumexp(x: paddle.Tensor, axis: int = -1) -> paddle.Tensor:
    m = paddle.max(x, axis=axis, keepdim=True)
    return (m + paddle.log(paddle.sum(paddle.exp(x - m), axis=axis, keepdim=True))).squeeze(axis)


class LinearChainCRF(paddle.nn.Layer):
    """Linear-chain CRF.

    Parameters
    ----------
    n_features : int
        Size of the per-token feature vector.
    n_tags : int
        Number of output tags.
    """

    def __init__(self, n_features: int, n_tags: int) -> None:
        super().__init__()
        if n_features <= 0 or n_tags <= 1:
            raise ValueError("n_features must be > 0 and n_tags >= 2")
        self.n_features = n_features
        self.n_tags = n_tags
        self.emitter = paddle.nn.Linear(n_features, n_tags)
        self.transitions = paddle.create_parameter(
            shape=[n_tags, n_tags], dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.1, 0.1),
        )
        self.start_transitions = paddle.create_parameter(
            shape=[n_tags], dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.1, 0.1),
        )
        self.end_transitions = paddle.create_parameter(
            shape=[n_tags], dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.1, 0.1),
        )

    def _emit_scores(self, features: paddle.Tensor) -> paddle.Tensor:
        """``[T, n_tags]``."""
        return self.emitter(features)

    def _forward_algorithm(
        self, emit: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Returns ``(log_Z, alpha, beta)`` for the marginal pass."""
        T = emit.shape[0]
        # alpha[t, k] = log-sum-exp over all paths that end in tag k at t.
        alpha = [self.start_transitions + emit[0]]              # [n_tags]
        for t in range(1, T):
            score = alpha[-1].unsqueeze(-1) + self.transitions  # [n_tags, n_tags]
            alpha.append(
                _logsumexp(score, axis=0) + emit[t]
            )
        alpha = paddle.stack(alpha, axis=0)                      # [T, n_tags]
        # beta[t, k] = log-sum-exp over all paths that start at t in tag k.
        beta = [self.end_transitions]                            # [n_tags]
        for t in range(T - 2, -1, -1):
            score = (
                self.transitions
                + emit[t + 1].unsqueeze(0)
                + beta[0].unsqueeze(0)
            )                                                   # [n_tags, n_tags]
            beta.insert(0, _logsumexp(score, axis=1))
        beta = paddle.stack(beta, axis=0)                        # [T, n_tags]
        log_Z = _logsumexp(alpha[T - 1] + self.end_transitions, axis=-1)
        return log_Z, alpha, beta

    def forward(self, features: paddle.Tensor) -> paddle.Tensor:
        """Return per-tag marginal posteriors, shape ``[T, n_tags]``."""
        emit = self._emit_scores(features)
        _, alpha, beta = self._forward_algorithm(emit)
        log_gamma = alpha + beta - _logsumexp(alpha[-1] + self.end_transitions, axis=-1)
        return paddle.exp(log_gamma)

    def nll(self, features: paddle.Tensor, tags: paddle.Tensor) -> paddle.Tensor:
        """Negative log-likelihood for training. ``tags`` is ``[T]`` int64."""
        emit = self._emit_scores(features)
        T = emit.shape[0]
        log_Z, _, _ = self._forward_algorithm(emit)
        # Score of the gold path.
        score = self.start_transitions[tags[0]] + emit[0, tags[0]]
        for t in range(1, T):
            score = score + self.transitions[tags[t - 1], tags[t]] + emit[t, tags[t]]
        score = score + self.end_transitions[tags[T - 1]]
        return log_Z - score

    def decode(self, features: paddle.Tensor) -> paddle.Tensor:
        """Viterbi MAP decode, returns int64 ``[T]``."""
        emit = self._emit_scores(features)
        T = emit.shape[0]
        V = [self.start_transitions + emit[0]]
        ptr = []
        for t in range(1, T):
            scores = V[-1].unsqueeze(-1) + self.transitions       # [k, l]
            best = paddle.max(scores, axis=0)                     # [l]
            idx = paddle.argmax(scores, axis=0)                    # [l]
            V.append(best + emit[t])
            ptr.append(idx)
        ptr = paddle.stack(ptr, axis=0) if ptr else paddle.zeros(
            [0, self.n_tags], dtype="int64"
        )
        best_last = int(paddle.argmax(V[-1] + self.end_transitions).numpy())
        path = [best_last]
        for t in range(T - 1, 0, -1):
            prev = int(ptr[t - 1, path[-1]].numpy())
            path.append(prev)
        path.reverse()
        return paddle.to_tensor(path, dtype="int64")

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, n_tags={self.n_tags}"
