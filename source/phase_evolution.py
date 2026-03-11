import torch

from compute_energy import gradients, positive_strain_energy_density


def _element_average(field, T_conn):
    return (field[T_conn[:, 0]] + field[T_conn[:, 1]] + field[T_conn[:, 2]]) / 3.0


def laplace_field(inp, field, area_elem, T_conn=None):
    """
    Computes Laplace(field).
    T_conn = None: use autodiff for second derivatives.
    T_conn != None: use a connectivity-based discrete Laplacian on nodes, then
    average it to element values.
    """
    if T_conn is None:
        grad_field = torch.autograd.grad(field.sum(), inp, create_graph=True)[0]
        grad_x = grad_field[:, 0]
        grad_y = grad_field[:, 1]

        grad_x_x = torch.autograd.grad(grad_x.sum(), inp, create_graph=True)[0][:, 0]
        grad_y_y = torch.autograd.grad(grad_y.sum(), inp, create_graph=True)[0][:, 1]
        return grad_x_x + grad_y_y

    n_nodes = field.shape[0]
    tri = T_conn.long()

    edges = torch.cat(
        [tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]],
        dim=0,
    )
    edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)
    src = edges[:, 0]
    dst = edges[:, 1]

    x = inp[:, 0]
    y = inp[:, 1]
    dx = x[src] - x[dst]
    dy = y[src] - y[dst]
    dist2 = dx * dx + dy * dy + torch.finfo(field.dtype).eps
    w = 1.0 / dist2

    diff = field[dst] - field[src]
    num = torch.zeros(n_nodes, dtype=field.dtype, device=field.device)
    den = torch.zeros(n_nodes, dtype=field.dtype, device=field.device)
    num.index_add_(0, src, w * diff)
    den.index_add_(0, src, w)
    lap_nodes = num / (den + torch.finfo(field.dtype).eps)

    return _element_average(lap_nodes, T_conn)


def compute_history_drive(inp, u, v, d_curr, matprop, pffmodel, area_elem, T_conn=None):
    """
    Computes the positive strain-energy density H_e candidate.
    This maps directly to the positive part returned by
    ``strain_energy_with_split(...)`` in compute_energy.py.
    """
    strain_11, strain_22, strain_12, _, _ = gradients(inp, u, v, d_curr, area_elem, T_conn)

    if T_conn is None:
        d_elem = d_curr
    else:
        d_elem = _element_average(d_curr, T_conn)

    return positive_strain_energy_density(strain_11, strain_22, strain_12, d_elem, matprop, pffmodel)


def compute_phase_evolution_residual(inp, d_curr, d_prev, dt, eta_pf, GcI, l0, H_e, area_elem, T_conn=None):
    lap_d = laplace_field(inp, d_curr, area_elem, T_conn)

    if T_conn is None:
        d_eval = d_curr
        d_prev_eval = d_prev
        H_eval = H_e
    else:
        d_eval = _element_average(d_curr, T_conn)
        d_prev_eval = _element_average(d_prev, T_conn)
        H_eval = H_e if H_e.shape[0] == d_eval.shape[0] else _element_average(H_e, T_conn)

    r = eta_pf * (d_eval - d_prev_eval) / dt - GcI * l0 * lap_d + (GcI / l0 + 2.0 * H_eval) * d_eval - 2.0 * H_eval
    return r


def compute_phase_evolution_loss(inp, d_curr, d_prev, dt, eta_pf, GcI, l0, H_e, area_elem,
                                 T_conn=None, reduction='mean'):
    r = compute_phase_evolution_residual(inp, d_curr, d_prev, dt, eta_pf, GcI, l0, H_e, area_elem, T_conn)

    if reduction == 'mean':
        return torch.mean(r ** 2)
    if reduction == 'area':
        return torch.sum(area_elem * (r ** 2))

    raise ValueError("reduction must be 'mean' or 'area'")
