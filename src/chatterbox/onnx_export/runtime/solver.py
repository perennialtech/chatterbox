import numpy as np


def meanflow_euler(session, noise, mu, mask, spks, cond, t_span):
    x = noise
    batch = x.shape[0]
    for t, r in zip(t_span[:-1], t_span[1:]):
        t_arr = np.full((batch,), t, dtype=x.dtype)
        r_arr = np.full((batch,), r, dtype=x.dtype)
        (dxdt,) = session.run(
            None,
            {
                "x": x,
                "mask": mask,
                "mu": mu,
                "spks": spks,
                "cond": cond,
                "t": t_arr,
                "r": r_arr,
            },
        )
        x = x + (r - t) * dxdt
    return x
