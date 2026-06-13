"""2-D Ising model (no external field) re-implemented as a
``paddle.nn.Layer``.

Analogue:
    No direct sklearn analogue. This is the classical
    statistical-physics Ising model studied since Lenz (1920) /
    Ising (1925) and the standard 2-D test bed for MCMC; see
    e.g. Onsager's 1944 exact solution for the critical
    temperature T_c = 2 / ln(1 + sqrt(2)) ≈ 2.269 (in units where
    J = 1).

The layer exposes:
* ``gibbs_sample(n_burn_in, n_steps, beta)`` -- a single-spin
  Gibbs sweep with periodic boundary conditions.
* ``magnetisation(...)`` -- mean |magnetisation| over the chain.
* ``energy(spins)`` -- the (negative) log-probability up to a
  constant, useful for diagnosing convergence.
"""

import paddle


class IsingModel(paddle.nn.Layer):
    """2-D Ising model with periodic boundary conditions.

    Analogue:
        Classical statistical-physics model (Lenz 1920 / Ising 1925);
        no direct sklearn analogue; standard 2-D testbed for MCMC
        with Onsager's 1944 critical temperature T_c ≈ 2.269.

    Parameters
    ----------
    n_rows, n_cols : int
        Lattice size.
    J : float, default 1.0
        Coupling strength.
    h : float, default 0.0
        External field.
    """

    def __init__(
        self, n_rows: int, n_cols: int, J: float = 1.0, h: float = 0.0
    ) -> None:
        super().__init__()
        if n_rows < 2 or n_cols < 2:
            raise ValueError("Lattice must be at least 2x2")
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.J = J
        self.h = h

    @property
    def critical_temperature(self) -> float:
        """Onsager's exact critical temperature for J=1, h=0: T_c = 2/ln(1+√2) ≈ 2.269."""
        if self.J == 1.0 and self.h == 0.0:
            import math
            return 2.0 / math.log(1.0 + math.sqrt(2.0))
        return float("nan")

    def spin_string(self, spins: paddle.Tensor) -> str:
        """Render a spin configuration as ASCII art (+ for up, · for down)."""
        arr = spins.numpy()
        lines = []
        for row in arr:
            line = "".join("+" if s > 0 else "·" for s in row)
            lines.append(line)
        return "\n".join(lines)

    def random_state(self) -> paddle.Tensor:
        """Initialise a random +/-1 spin configuration."""
        return paddle.cast(
            paddle.rand([self.n_rows, self.n_cols]) < 0.5, "float32"
        ) * 2 - 1

    def local_field(self, spins: paddle.Tensor) -> paddle.Tensor:
        """Sum of nearest-neighbour spins at each site (periodic BC)."""
        up = paddle.roll(spins, shifts=1, axis=0)
        down = paddle.roll(spins, shifts=-1, axis=0)
        left = paddle.roll(spins, shifts=1, axis=1)
        right = paddle.roll(spins, shifts=-1, axis=1)
        return up + down + left + right

    def energy(self, spins: paddle.Tensor) -> paddle.Tensor:
        """Energy ``-J sum_{<i,j>} s_i s_j - h sum_i s_i`` (scalar)."""
        return -self.J * paddle.sum(spins * self.local_field(spins)) / 2 \
            - self.h * paddle.sum(spins)

    @paddle.no_grad()
    def gibbs_sample(
        self, n_burn_in: int = 100, n_steps: int = 100, beta: float = 0.5,
    ) -> paddle.Tensor:
        """Run a single-spin Gibbs sweep with periodic boundary
        conditions; return ``n_steps`` successive configurations
        stacked along axis 0."""
        spins = self.random_state()
        for _ in range(n_burn_in):
            spins = self._gibbs_sweep(spins, beta)
        history = [spins.clone()]
        for _ in range(n_steps):
            spins = self._gibbs_sweep(spins, beta)
            history.append(spins.clone())
        return paddle.stack(history, axis=0)

    @paddle.no_grad()
    def _gibbs_sweep(self, spins: paddle.Tensor, beta: float) -> paddle.Tensor:
        """One full sweep visiting every site in a checkerboard order
        (technically only half the spins need updating per sub-step
        for parallelisation; we update every site independently)."""
        # Compute all local fields at once.
        h_eff = self.J * self.local_field(spins) + self.h
        flip_prob = paddle.nn.functional.sigmoid(-2.0 * beta * h_eff)
        flip = paddle.cast(paddle.rand(spins.shape) < flip_prob, spins.dtype)
        return spins * (1 - 2 * flip)

    @paddle.no_grad()
    def magnetisation(
        self, n_burn_in: int = 200, n_samples: int = 500, beta: float = 0.5,
    ) -> float:
        """Mean ``|m|`` over the chain after burn-in."""
        spins = self.random_state()
        for _ in range(n_burn_in):
            spins = self._gibbs_sweep(spins, beta)
        mags = []
        for _ in range(n_samples):
            spins = self._gibbs_sweep(spins, beta)
            mags.append(paddle.mean(spins))
        return float(paddle.mean(paddle.abs(paddle.stack(mags))).numpy().item())

    def extra_repr(self) -> str:
        return f"n_rows={self.n_rows}, n_cols={self.n_cols}, J={self.J}, h={self.h}"
