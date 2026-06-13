"""GMM-HMM (Rabiner 1989) re-implemented as a ``paddle.nn.Layer``.

Each hidden state ``k`` has its own Gaussian Mixture Model over the
observation space, parameterised by

* ``means``         [n_states, n_components, n_features]
* ``log_vars``      [n_states, n_components, n_features]   (diagonal)
* ``log_weights``   [n_states, n_components]
* ``log_start``     [n_states]
* ``log_trans``     [n_states, n_states]

The forward pass returns per-timestep responsibilities of the
hidden state, and ``viterbi`` returns the most-likely state
sequence. ``fit_em`` performs Baum-Welch EM with closed-form
M-step for the GMM emission parameters.
"""

import paddle


def _logsumexp(x: paddle.Tensor, axis: int = -1, keepdim: bool = False) -> paddle.Tensor:
    m = paddle.max(x, axis=axis, keepdim=True)
    out = m + paddle.log(paddle.sum(paddle.exp(x - m), axis=axis, keepdim=True))
    if not keepdim:
        out = out.squeeze(axis)
    return out


class GMMHMM(paddle.nn.Layer):
    """GMM-emission HMM.

    Parameters
    ----------
    n_states : int
    n_components : int
        Number of Gaussian components per state.
    n_features : int
        Observation dimension.
    """

    def __init__(self, n_states: int, n_components: int, n_features: int) -> None:
        super().__init__()
        if n_states < 2 or n_components < 1 or n_features < 1:
            raise ValueError("n_states>=2, n_components>=1, n_features>=1")
        self.n_states = n_states
        self.n_components = n_components
        self.n_features = n_features

        self.log_start = paddle.create_parameter(
            shape=[n_states], dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.5, 0.5),
        )
        self.log_trans = paddle.create_parameter(
            shape=[n_states, n_states], dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.5, 0.5),
        )
        self.means = paddle.create_parameter(
            shape=[n_states, n_components, n_features], dtype="float32",
            default_initializer=paddle.nn.initializer.Uniform(-0.5, 0.5),
        )
        self.log_vars = paddle.create_parameter(
            shape=[n_states, n_components, n_features], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )
        self.log_comp_weights = paddle.create_parameter(
            shape=[n_states, n_components], dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

    def _start(self) -> paddle.Tensor:
        return paddle.nn.functional.softmax(self.log_start, axis=-1)

    def _trans(self) -> paddle.Tensor:
        return paddle.nn.functional.softmax(self.log_trans, axis=-1)

    # ---------------------------------------------- emission log-probs
    def _log_emission(self) -> paddle.Tensor:
        """Return ``log p(y | state=k, mixture=c)`` evaluated at an
        empty placeholder. We instead compute log p per datum on
        the fly inside ``_log_emit_x``."""
        return None  # computed on the fly

    def _log_emit_x(self, x: paddle.Tensor) -> paddle.Tensor:
        """Log emission probabilities for sequence ``x: [T, n_features]``.

        Returns ``[T, n_states]`` log p(x_t | state=k).
        """
        # log p(x | state=k, comp=c) is sum_d -0.5 ((x_d - mu_cd)^2 / var_cd + log var + log 2π)
        # Vectorise: for each state k,
        #   log p(x | k) = logsumexp_c [ log_w[k, c] + sum_d log_gauss_cd(x) ].
        T = x.shape[0]
        # x: [T, F] -> [T, 1, 1, F]
        x4 = x.unsqueeze(1).unsqueeze(1)
        # means, log_vars, log_comp_weights: [K, C, F]
        diff = x4 - self.means.unsqueeze(0)
        var = paddle.exp(self.log_vars.unsqueeze(0))
        # Per-component Gaussian log-density
        log_p_cd = -0.5 * (diff ** 2 / var) - 0.5 * self.log_vars.unsqueeze(0) \
            - 0.5 * paddle.log(paddle.to_tensor(2 * 3.141592653589793, dtype=x.dtype))
        # Sum over features
        log_p_c = log_p_cd.sum(axis=-1)                          # [T, K, C]
        # Add log component weights
        log_p_c = log_p_c + paddle.nn.functional.log_softmax(
            self.log_comp_weights, axis=-1
        ).unsqueeze(0)
        # Logsumexp over components
        return _logsumexp(log_p_c, axis=-1)                      # [T, K]

    # ---------------------------------------------- forward-backward
    def _forward_backward(
        self, log_emit_x: paddle.Tensor
    ) -> tuple:
        T = log_emit_x.shape[0]
        log_start = paddle.log(self._start() + 1e-30)
        log_trans = paddle.log(self._trans() + 1e-30)

        log_alpha = [log_start + log_emit_x[0]]
        for t in range(1, T):
            prev = log_alpha[-1].unsqueeze(-1)
            log_alpha.append(_logsumexp(prev + log_trans, axis=0) + log_emit_x[t])
        log_alpha = paddle.stack(log_alpha, axis=0)

        log_beta = [paddle.zeros([self.n_states], dtype=log_alpha.dtype)]
        for t in range(T - 2, -1, -1):
            post = log_beta[0]
            log_beta.insert(
                0,
                _logsumexp(
                    log_trans + post.unsqueeze(0) + log_emit_x[t + 1].unsqueeze(0),
                    axis=1,
                ),
            )
        log_beta = paddle.stack(log_beta, axis=0)

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - _logsumexp(log_gamma, axis=-1, keepdim=True)
        gamma = paddle.exp(log_gamma)
        log_lik = float(_logsumexp(log_alpha[T - 1], axis=-1).numpy().item())

        xi_list = []
        for t in range(T - 1):
            log_xi = (
                log_alpha[t].unsqueeze(-1)
                + log_trans
                + log_emit_x[t + 1].unsqueeze(0)
                + log_beta[t + 1].unsqueeze(0)
            )
            log_xi = log_xi - _logsumexp(log_xi, axis=-1, keepdim=True)
            xi_list.append(paddle.exp(log_xi))
        xi = paddle.stack(xi_list, axis=0) if xi_list else paddle.zeros(
            [0, self.n_states, self.n_states], dtype=log_alpha.dtype
        )
        return gamma, xi, log_lik

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D sequence, got {x.shape}")
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected n_features={self.n_features}, got {x.shape[1]}"
            )
        log_emit_x = self._log_emit_x(x)
        gamma, _, _ = self._forward_backward(log_emit_x)
        return gamma

    def viterbi(self, x: paddle.Tensor) -> paddle.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        log_emit_x = self._log_emit_x(x)
        T = x.shape[0]
        log_start = paddle.log(self._start() + 1e-30)
        log_trans = paddle.log(self._trans() + 1e-30)
        V = [log_start + log_emit_x[0]]
        ptr = []
        for t in range(1, T):
            scores = V[-1].unsqueeze(-1) + log_trans
            best = paddle.max(scores, axis=0)
            idx = paddle.argmax(scores, axis=0)
            V.append(best + log_emit_x[t])
            ptr.append(idx)
        ptr = paddle.stack(ptr, axis=0) if ptr else paddle.zeros(
            [0, self.n_states], dtype="int64"
        )
        best_last = int(paddle.argmax(V[-1]).numpy())
        path = [best_last]
        for t in range(T - 1, 0, -1):
            prev = int(ptr[t - 1, path[-1]].numpy())
            path.append(prev)
        path.reverse()
        return paddle.to_tensor(path, dtype="int64")

    # ------------------------------------------------------------------- EM
    @paddle.no_grad()
    def fit_em(self, x: paddle.Tensor, n_iter: int = 30) -> "GMMHMM":
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        for _ in range(n_iter):
            log_emit_x = self._log_emit_x(x)
            gamma, xi, _ = self._forward_backward(log_emit_x)
            # M-step for start, trans (same as GaussianHMM)
            new_start = gamma[0]
            xi_sum = xi.sum(axis=0)
            gamma_sum = paddle.clip(gamma[:-1].sum(axis=0), min=1e-12)
            new_trans = xi_sum / gamma_sum.unsqueeze(-1)
            new_trans = new_trans / paddle.clip(
                new_trans.sum(axis=1, keepdim=True), min=1e-12
            )

            # M-step for the GMM emission parameters using the
            # per-state responsibilities.
            # gamma_t: [T, K], x: [T, F]
            T, K, C, F = self.n_states, self.n_states, self.n_components, self.n_features
            new_means = paddle.zeros([K, C, F], dtype=x.dtype)
            new_vars = paddle.zeros([K, C, F], dtype=x.dtype)
            new_comp = paddle.zeros([K, C], dtype=x.dtype)
            for k in range(K):
                gk = gamma[:, k]                                  # [T]
                # gk_total / (sum_c gk_total * log_w[c]) gives a soft
                # assignment of each (t, c) to state k.
                # First compute log_emit_per_comp: [T, C]
                diff = x.unsqueeze(1) - self.means[k]             # [T, C, F]
                var = paddle.exp(self.log_vars[k])
                log_p_cd = -0.5 * (diff ** 2 / var) - 0.5 * self.log_vars[k] \
                    - 0.5 * paddle.log(paddle.to_tensor(2 * 3.141592653589793, dtype=x.dtype))
                log_p_c = log_p_cd.sum(axis=-1) + paddle.nn.functional.log_softmax(
                    self.log_comp_weights[k], axis=-1
                )                                                # [T, C]
                log_p_c = log_p_c - _logsumexp(log_p_c, axis=-1, keepdim=True)
                p_c = paddle.exp(log_p_c)                         # [T, C]
                # weight of (t, c) into state k
                w = gk.unsqueeze(-1) * p_c                        # [T, C]
                sum_w = w.sum(axis=0) + 1e-12                     # [C]
                new_means[k] = (w.unsqueeze(-1) * x.unsqueeze(1)).sum(axis=0) / sum_w.unsqueeze(-1)
                diff2 = (x.unsqueeze(1) - new_means[k]) ** 2
                new_vars[k] = (w.unsqueeze(-1) * diff2).sum(axis=0) / sum_w.unsqueeze(-1) + 1e-4
                new_comp[k] = sum_w / (sum_w.sum() + 1e-12)

            self.log_start.set_value(paddle.log(new_start + 1e-30))
            self.log_trans.set_value(paddle.log(new_trans + 1e-30))
            self.means.set_value(new_means)
            self.log_vars.set_value(paddle.log(new_vars + 1e-12))
            self.log_comp_weights.set_value(paddle.log(new_comp + 1e-30))
        return self

    def extra_repr(self) -> str:
        return (
            f"n_states={self.n_states}, n_components={self.n_components}, "
            f"n_features={self.n_features}"
        )
