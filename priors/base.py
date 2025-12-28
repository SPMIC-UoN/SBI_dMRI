"""
priors/base.py

Low-level prior distributions and helpers used across models.

This module intentionally contains:
- Generic, reusable distributions (e.g., angle priors, ARD prior)
- Small helper constructors (BoxUniform/Gamma convenience wrappers)

It should NOT contain model-specific Ω assembly logic (that belongs in
priors/<model>.py, e.g. priors/ball_and_sticks.py).
"""

from __future__ import annotations

import math
import torch
from torch.distributions import Distribution, Uniform, Gamma
from sbi.utils.torchutils import BoxUniform


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def box_uniform_1d(low: float, high: float, device: str = "cpu") -> BoxUniform:
    """
    Convenience wrapper: 1D BoxUniform with event shape (1,).

    Notes
    -----
    - Returns samples with shape (..., 1).
    - We wrap scalars as 1D tensors because sbi's BoxUniform expects tensors.
    """
    return BoxUniform(
        low=torch.as_tensor([low], device=device, dtype=torch.float32),
        high=torch.as_tensor([high], device=device, dtype=torch.float32),
    )


def gamma_1d(shape: float, rate: float, device: str = "cpu") -> Gamma:
    """
    Convenience wrapper: 1D Gamma distribution.

    Parameters
    ----------
    shape : float
        Gamma shape (k / alpha).
    rate : float
        Gamma rate (1/scale) (beta).
    """
    return Gamma(
        torch.tensor([shape], device=device, dtype=torch.float32),
        torch.tensor([rate], device=device, dtype=torch.float32),
    )


# -----------------------------------------------------------------------------
# Orientation priors (theta, phi)
# -----------------------------------------------------------------------------
class CorrectedThetaUniform(Distribution):
    """
    Polar angle prior for fibre orientation: theta ∈ [0, π].

    Sampling
    --------
    Let u ~ Uniform(0, 1). We set:
        theta = arccos(1 - 2u)

    This makes cos(theta) uniform on [-1, 1], which is the correct construction
    for *uniform directions on the sphere* when combined with an independent
    azimuthal angle phi uniform on [0, 2π).

    Sphere vs hemisphere
    --------------------
    - Theta is ALWAYS sampled in [0, π] (full polar range).
    - Whether you cover the full sphere or only a hemisphere is controlled by
      the phi distribution:
        * phi ~ Uniform(0, 2π)  -> full sphere directions
        * phi ~ Uniform(0,  π)  -> hemisphere restriction (antipodal ambiguity)

    Density
    -------
    For uniform directions on the sphere, the marginal density of theta is:
        p(theta) = 0.5 * sin(theta),  theta ∈ [0, π].
    """
    has_rsample = False
    arg_constraints = {}  # silence torch warning

    def __init__(self, device: str = "cpu", eps: float = 1e-12):
        super().__init__()
        self._u = box_uniform_1d(0.0, 1.0, device=device)
        self._batch_shape = self._u.batch_shape
        self._event_shape = self._u.event_shape
        self._eps = eps

    def sample(self, sample_shape=torch.Size()):
        u = self._u.sample(sample_shape)  # (..., 1)
        return torch.acos(1.0 - 2.0 * u)  # (..., 1)

    def log_prob(self, value):
        """
        log p(theta) = log(0.5) + log(sin(theta)) for theta in [0, π],
        else -inf.

        Note: we clamp sin(theta) away from 0 to avoid log(0) numerical issues.
        """
        theta = torch.as_tensor(value)
        # support mask
        in_support = (theta >= 0.0) & (theta <= math.pi)
        sin_theta = torch.sin(theta).clamp_min(self._eps)

        logp = math.log(0.5) + torch.log(sin_theta)
        logp = torch.where(in_support, logp, torch.tensor(float("-inf"), device=theta.device))
        # BoxUniform returns event shape (1,), keep same
        return logp

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape


class PhiUniformHemisphere(Distribution):
    """
    Azimuthal angle prior for hemisphere restriction: phi ∈ [0, π].

    Using phi ∈ [0, π] (instead of [0, 2π]) restricts orientations to a
    hemisphere. This is commonly done to remove antipodal ambiguity in fibre
    orientation (v and -v represent the same axis in many dMRI models).

    Combined with CorrectedThetaUniform (theta ∈ [0, π]), this covers a hemisphere.

    Density
    -------
        p(phi) = 1/π,   phi ∈ [0, π].
    """
    has_rsample = False
    arg_constraints = {}  # silence torch warning

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self._u = box_uniform_1d(0.0, 1.0, device=device)
        self._batch_shape = self._u.batch_shape
        self._event_shape = self._u.event_shape

    def sample(self, sample_shape=torch.Size()):
        u = self._u.sample(sample_shape)
        return math.pi * u  # phi in [0, pi]

    def log_prob(self, value):
        phi = torch.as_tensor(value)
        in_support = (phi >= 0.0) & (phi <= math.pi)
        logp = torch.full_like(phi, fill_value=-math.log(math.pi))
        logp = torch.where(in_support, logp, torch.tensor(float("-inf"), device=phi.device))
        return logp

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape


class PhiUniformSphere(Distribution):
    """
    Azimuthal angle prior for full-sphere coverage: phi ∈ [0, 2π].

    Combined with CorrectedThetaUniform (theta ∈ [0, π]), this produces
    directions uniformly distributed on the full sphere.

    Density
    -------
        p(phi) = 1/(2π),   phi ∈ [0, 2π].
    """
    has_rsample = False
    arg_constraints = {}  # silence torch warning

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self._u = box_uniform_1d(0.0, 1.0, device=device)
        self._batch_shape = self._u.batch_shape
        self._event_shape = self._u.event_shape

    def sample(self, sample_shape=torch.Size()):
        u = self._u.sample(sample_shape)
        return 2.0 * math.pi * u  # phi in [0, 2pi]

    def log_prob(self, value):
        phi = torch.as_tensor(value)
        in_support = (phi >= 0.0) & (phi <= 2.0 * math.pi)
        logp = torch.full_like(phi, fill_value=-math.log(2.0 * math.pi))
        logp = torch.where(in_support, logp, torch.tensor(float("-inf"), device=phi.device))
        return logp

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape


# -----------------------------------------------------------------------------
# Other generic priors
# -----------------------------------------------------------------------------
class ARDPrior(Distribution):
    """
    Approximate ARD prior: p(x) ∝ 1/x on a bounded domain [min_val, max_val].

    Sampling is performed via log-uniform:
        log(x) ~ Uniform(log(min_val), log(max_val))

    Notes
    -----
    - This is a bounded approximation to the improper 1/x prior.
    - Included here because it's generic and can be used in multiple models.
    """
    has_rsample = False
    arg_constraints = {}  # silence torch warning

    def __init__(self, min_val: float = 1e-5, max_val: float = 0.99, device: str = "cpu"):
        super().__init__()
        self.min_val = torch.tensor([min_val], device=device, dtype=torch.float32)
        self.max_val = torch.tensor([max_val], device=device, dtype=torch.float32)

        self._batch_shape = torch.Size([1])
        self._event_shape = torch.Size([])

        self._uniform = Uniform(torch.log(self.min_val), torch.log(self.max_val))

    def sample(self, sample_shape=torch.Size()):
        return torch.exp(self._uniform.sample(sample_shape))

    def log_prob(self, value):
        value = torch.as_tensor(value)
        logp = -torch.log(value)
        logp = torch.where(
            (value < self.min_val) | (value > self.max_val),
            torch.tensor(float("-inf"), device=value.device),
            logp,
        )
        return logp

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape