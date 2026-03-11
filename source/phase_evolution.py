import torch


def compute_phase_evolution_loss(d, d_prev, dt, eta_pf, GcI):
    '''
    Computes a viscous-time phase evolution residual penalty.
    '''
    if d_prev is None:
        raise ValueError('d_prev must be provided when phase_mode is "viscous_time"')
    if dt is None:
        raise ValueError('dt must be provided when phase_mode is "viscous_time"')
    if eta_pf is None:
        raise ValueError('eta_pf must be provided when phase_mode is "viscous_time"')
    if GcI is None:
        raise ValueError('GcI must be provided when phase_mode is "viscous_time"')

    dt_safe = dt if torch.is_tensor(dt) else torch.tensor(dt, dtype=d.dtype, device=d.device)
    eta_pf_t = eta_pf if torch.is_tensor(eta_pf) else torch.tensor(eta_pf, dtype=d.dtype, device=d.device)
    GcI_t = GcI if torch.is_tensor(GcI) else torch.tensor(GcI, dtype=d.dtype, device=d.device)

    rate = (d - d_prev)/(dt_safe + torch.finfo(d.dtype).eps)
    residual = eta_pf_t*rate/GcI_t
    return torch.mean(residual**2)
