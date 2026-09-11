"""Adam's update rule in plain Python — no torch in the math path.

The whole point of Experiments 1 and 2 is that every intermediate number is
visible, so this module deals in Python ``float`` only. See
``experiments/exp1_adam_by_hand.py`` for a worked example and
``tests/test_adam_matches_torch.py`` for the assertion that this matches
``torch.optim.Adam`` to nine decimal places.
"""

from __future__ import annotations


def adam_step(
    w: float,
    g: float,
    m: float,
    v: float,
    t: int,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
    bias_correction: bool = True,
) -> tuple[float, float, float, float, float]:
    """Run one Adam update on a single scalar weight.

    Parameters
    ----------
    w : current weight value.
    g : gradient (the "hint") for this step.
    m : first-moment memory (running average of gradients) from the previous step.
    v : second-moment memory (running average of squared gradients) from the previous step.
    t : step number, starting at 1 (used only by the bias correction).
    lr, b1, b2, eps : the usual Adam hyperparameters (learning rate, beta1, beta2, epsilon).
    bias_correction : when False, m_hat/v_hat are left uncorrected (Experiment 2).

    Returns
    -------
    (w_new, m_new, v_new, m_hat, v_hat)

    The actual step taken is ``w - w_new`` and equals
    ``lr * m_hat / (sqrt(v_hat) + eps)``. This matches PyTorch's Adam, which
    computes ``step_size = lr / (1 - b1**t)`` and
    ``denom = sqrt(v) / sqrt(1 - b2**t) + eps``; the two formulations are
    algebraically identical.
    """
    # Update the two running memories.
    m_new = b1 * m + (1.0 - b1) * g
    v_new = b2 * v + (1.0 - b2) * (g * g)

    if bias_correction:
        m_hat = m_new / (1.0 - b1**t)
        v_hat = v_new / (1.0 - b2**t)
    else:
        m_hat = m_new
        v_hat = v_new

    update = lr * m_hat / (v_hat**0.5 + eps)
    w_new = w - update
    return w_new, m_new, v_new, m_hat, v_hat
