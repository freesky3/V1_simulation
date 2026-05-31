from __future__ import annotations


def make_transfer_interpolator(jnp):
    def interp_phi(x, xp, fp, rate_max):
        out = jnp.interp(x, xp, fp, left=fp[0], right=fp[-1])
        return jnp.where(jnp.isfinite(rate_max), jnp.clip(out, 0.0, rate_max), out)

    return interp_phi


def make_wilson_cowan_jax_rhs(jnp, *, is_static: bool):
    """Build the shared JAX Wilson-Cowan RHS used by RK4 and Diffrax kernels."""
    interp_phi = make_transfer_interpolator(jnp)

    def rhs(
        y,
        ax,
        bg_e,
        bg_i,
        W_exc,
        W_inh,
        W_ext,
        mu_ext,
        idx_exc,
        idx_inh,
        phi_exc_x,
        phi_exc_y,
        phi_exc_rate_max,
        phi_inh_x,
        phi_inh_y,
        phi_inh_rate_max,
        tau_exc,
        tau_inh,
    ):
        if is_static:
            mu = W_exc @ y[idx_exc, :] + W_inh @ y[idx_inh, :] + mu_ext
        else:
            mu = W_exc @ y[idx_exc, :] + W_inh @ y[idx_inh, :] + W_ext @ ax

        dy = jnp.zeros_like(y)
        dy = dy.at[idx_exc, :].set(
            (
                -y[idx_exc, :]
                + interp_phi(
                    tau_exc * mu[idx_exc, :] + bg_e,
                    phi_exc_x,
                    phi_exc_y,
                    phi_exc_rate_max,
                )
            )
            / tau_exc
        )
        dy = dy.at[idx_inh, :].set(
            (
                -y[idx_inh, :]
                + interp_phi(
                    tau_inh * mu[idx_inh, :] + bg_i,
                    phi_inh_x,
                    phi_inh_y,
                    phi_inh_rate_max,
                )
            )
            / tau_inh
        )
        return dy

    return rhs
