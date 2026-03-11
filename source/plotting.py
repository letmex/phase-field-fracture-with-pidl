import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import cm
import torch
import numpy as np
import copy
from pathlib import Path

from compute_energy import gradients, stress, compute_energy
from utils import parse_mesh



def plot_mesh(mesh_file, figdir):
    X, Y, T, _ = parse_mesh(filename = mesh_file, gradient_type = 'numerical')
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.triplot(X, Y, T, color='black', linewidth=1, rasterized=True)
    ax.set_axis_off()
    plt.savefig(figdir["png"]/Path('mesh.png'), transparent=True, bbox_inches='tight', dpi=600)
    plt.savefig(figdir["pdf"]/Path('mesh.pdf'), transparent=True, bbox_inches='tight', dpi=600)


def plot_field(inp, field, T, figname, figdir, dpi=300):
    input_pt = copy.deepcopy(inp)
    input_pt = input_pt.detach().numpy()
    triang = T
    if T == None:
        triang = tri.Triangulation(input_pt[:, 0], input_pt[:, 1]).triangles
        figname = figname + '-at-gp'

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_aspect('equal')
    tpc0 = ax.tripcolor(input_pt[:, 0], input_pt[:, 1], triang, field, shading='gouraud', rasterized=True)
    cbar = fig.colorbar(tpc0, ax = ax)
    cbar.formatter.set_powerlimits((0, 0))
    ax.set_title(figname)
    plt.savefig(figdir["png"]/Path(str(figname)+'.png'), transparent=True, bbox_inches='tight', dpi=dpi)
    plt.savefig(figdir["pdf"]/Path(str(figname)+'.pdf'), transparent=True, bbox_inches='tight', dpi=dpi)


def plot_energy(field_comp, disp, pffmodel, matprop, inp, T_conn, area_elem, trainedModel_path, figdir):
    energy = np.zeros([1, 2])

    j = 0
    file_exists = True
    while file_exists:
        model = trainedModel_path/Path('trained_1NN_'+str(j)+'.pt')
        if not Path.is_file(model):
            break
        field_comp.net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
        field_comp.lmbda = torch.tensor(disp[j])
        if T_conn == None:
            inp.requires_grad = True
        u, v, d, T_field = field_comp.fieldCalculation(inp)
        E_el, E_d, _ = compute_energy(inp, u, v, d, d, matprop, pffmodel, area_elem, T_conn)
        E_el, E_d = E_el.detach().numpy(), E_d.detach().numpy()
        energy = np.append(energy, np.array([[E_el, E_d]]), axis = 0)
        j += 1

    if j>0:
        energy = np.delete(energy, 0, 0)

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(disp[0:j], energy[0:j, 0], '-', label=r'$\mathcal{E}^{el}_{\theta}$')
        ax.plot(disp[0:j], energy[0:j, 1], '-', label=r'$\mathcal{E}^{d}_{\theta}$')
        ax.set_xlim((disp[0], disp[j-1]))
        ax.set_ylim((np.min(energy), np.max(energy)*1.1))
        ax.set_xlabel(r'$U_p$')
        ax.set_ylabel(r'$\mathcal{E}$')
        ax.legend(loc=2)
        plt.savefig(figdir["png"]/Path('energy_Up.png'), transparent=True, bbox_inches='tight')
        plt.savefig(figdir["pdf"]/Path('energy_Up.pdf'), transparent=True, bbox_inches='tight')
    else:
        print(f"No trained network available in {trainedModel_path}")


def collect_phase_time_history(field_comp, disp, inp, trainedModel_path, T_conn=None, time=None):
    disp_hist = []
    phase_hist = []

    j = 0
    while True:
        model = trainedModel_path/Path('trained_1NN_'+str(j)+'.pt')
        if not Path.is_file(model):
            break

        field_comp.net.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
        field_comp.lmbda = torch.tensor(disp[j])
        if T_conn == None:
            inp.requires_grad = True
        _, _, d, _ = field_comp.fieldCalculation(inp)

        disp_hist.append(disp[j])
        phase_hist.append(torch.max(d).detach().item())
        j += 1

    if j == 0:
        print(f"No trained network available in {trainedModel_path}")
        return None

    time_hist = None
    if time is not None:
        time_hist = np.asarray(time)[0:j]

    return {
        "disp": np.asarray(disp_hist),
        "phase": np.asarray(phase_hist),
        "time": time_hist,
        "step": np.arange(j),
        # generic aliases for extensibility
        "disp_hist": np.asarray(disp_hist),
        "time_hist": time_hist if time_hist is not None else np.arange(j),
        "d": {"d_max": np.asarray(phase_hist)}
    }


def plot_phase_time_history(history, figdir, phase_mode="static"):
    if history is None:
        return

    def _save(fig, stem):
        plt.savefig(figdir["png"]/Path(f"{stem}.png"), transparent=True, bbox_inches='tight')
        plt.savefig(figdir["pdf"]/Path(f"{stem}.pdf"), transparent=True, bbox_inches='tight')
        plt.close(fig)

    # Backward-compatible simple schema: disp/phase/time/step
    if "phase" in history and "disp" in history:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(history["disp"], history["phase"], '-')
        ax.set_xlabel(r'$U_p$')
        ax.set_ylabel(r'$\max(d)$')
        ax.set_title('phase-history-载荷')
        _save(fig, 'phase-history_载荷')

        if phase_mode == "static":
            print("非真实时间积分，仅为步号/伪时间对齐")
            return

        fig, ax = plt.subplots(figsize=(3, 2))
        if history.get("time") is not None:
            x = history["time"]
            title = 'phase-history-时间'
            xlabel = 'time'
        else:
            x = history.get("step", np.arange(len(history["phase"])))
            title = 'phase-history-pseudo-time(step index)'
            xlabel = 'pseudo-time(step index)'
        ax.plot(x, history["phase"], '-')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\max(d)$')
        ax.set_title(title)
        _save(fig, 'phase-history_时间')
        return

    # Generic schema: disp_hist/time_hist + metric groups (d/T/H_e)
    def _as_array(values, key_name):
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"history['{key_name}'] must be a 1D sequence")
        return arr

    def _metric_token(group_name, metric_name):
        if group_name in ["d", "T"]:
            return metric_name.replace("_", "")
        if group_name == "H_e":
            return metric_name.replace("H_e", "He").replace("_", "")
        return metric_name.replace("_", "")

    def _plot_metric(x_values, y_values, group_name, metric_name, axis_suffix, xlabel):
        metric_token = _metric_token(group_name, metric_name)
        filename = f"phase_time_history_{metric_token}_vs_{axis_suffix}"

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x_values, y_values, "-o", linewidth=1.5, markersize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs {axis_suffix}")
        ax.grid(alpha=0.3)
        _save(fig, filename)

    if "disp_hist" not in history:
        raise KeyError("history must include 'disp_hist'")
    disp_hist = _as_array(history["disp_hist"], "disp_hist")

    has_real_time = "time_hist" in history and history["time_hist"] is not None
    time_hist = _as_array(history["time_hist"], "time_hist") if has_real_time else np.arange(len(disp_hist), dtype=float)

    if len(time_hist) != len(disp_hist):
        raise ValueError("history['time_hist'] length must match history['disp_hist'] length")

    required_groups = ["d", "T", "H_e"]
    groups = [g for g in required_groups if g in history and isinstance(history[g], dict) and len(history[g]) > 0]
    if not groups:
        raise ValueError("history must include at least one non-empty metric group among d/T/H_e")

    static_notice_printed = False
    for group_name in groups:
        for metric_name, metric_values in history[group_name].items():
            y_values = _as_array(metric_values, f"{group_name}.{metric_name}")
            if len(y_values) != len(disp_hist):
                raise ValueError(
                    f"history['{group_name}']['{metric_name}'] length ({len(y_values)}) "
                    f"must match history['disp_hist'] length ({len(disp_hist)})"
                )

            _plot_metric(
                disp_hist,
                y_values,
                group_name,
                metric_name,
                axis_suffix="load",
                xlabel="prescribed displacement"
            )

            if phase_mode == "static":
                if not static_notice_printed:
                    print("非真实时间积分，仅为步号/伪时间对齐")
                    static_notice_printed = True
                continue

            time_xlabel = "time" if has_real_time else "pseudo-time(step index)"
            _plot_metric(
                time_hist,
                y_values,
                group_name,
                metric_name,
                axis_suffix="time",
                xlabel=time_xlabel
            )


def img_plot(field_comp, pffmodel, matprop, inp, T, area_elem, figdir, dpi=300):
    if T == None:
        inp.requires_grad = True
    u, v, d, T_field = field_comp.fieldCalculation(inp)
    strain_11, strain_22, strain_12, grad_alpha_x, grad_alpha_y = gradients(inp, u, v, d, area_elem, T)

    if T == None:
        input_elem = inp
        alpha_elem = d
    else:    
        input_elem = (inp[T[:, 0], :] + inp[T[:, 1], :] + inp[T[:, 2], :])/3
        alpha_elem = (d[T[:, 0]] + d[T[:, 1]] + d[T[:, 2]])/3
    stress_11, stress_22, stress_12 = stress(strain_11, strain_22, strain_12, alpha_elem, matprop, pffmodel) 

    stress_1 = 0.5*(stress_11 + stress_22) + torch.sqrt((0.5*(stress_11 - stress_22))**2 + stress_12**2)
    stress_2 = 0.5*(stress_11 + stress_22) - torch.sqrt((0.5*(stress_11 - stress_22))**2 + stress_12**2)

    input_pt = copy.deepcopy(inp)
    input_el = copy.deepcopy(input_elem)
    input_pt, input_el = input_pt.detach().numpy(), input_el.detach().numpy()
    u, v, d, T_field = u.detach().numpy(), v.detach().numpy(), d.detach().numpy(), T_field.detach().numpy()
    strain_11, strain_22, strain_12 = strain_11.detach().numpy(), strain_22.detach().numpy(), strain_12.detach().numpy()
    stress_11, stress_22, stress_12 = stress_11.detach().numpy(), stress_22.detach().numpy(), stress_12.detach().numpy()
    stress_1, stress_2 = stress_1.detach().numpy(), stress_2.detach().numpy()
    disp = field_comp.lmbda.item()

    if T == None:
        x = input_pt[:, 0]
        y = input_pt[:, 1]
        T = tri.Triangulation(x, y).triangles

    fig, ax = plt.subplots(figsize=(9.5, 2), ncols=3)
    ax[0].set_aspect('equal')
    tpc0 = ax[0].tripcolor(input_pt[:, 0], input_pt[:, 1], T, u, shading='gouraud', rasterized=True)
    cbar0 = fig.colorbar(tpc0, ax = ax[0])
    cbar0.formatter.set_powerlimits((0, 0))
    ax[0].set_title(r"$u_{\theta}$")

    ax[1].set_aspect('equal')
    tpc1 = ax[1].tripcolor(input_pt[:, 0], input_pt[:, 1], T, v, shading='gouraud', rasterized=True)
    cbar1 = fig.colorbar(tpc1, ax = ax[1])
    cbar1.formatter.set_powerlimits((0, 0))
    ax[1].set_title(r"$v_{\theta}$")

    ax[2].set_aspect('equal')
    tpc2 = ax[2].tripcolor(input_pt[:, 0], input_pt[:, 1], T, d, shading='gouraud', rasterized=True)
    cbar2 = fig.colorbar(tpc2, ax = ax[2])
    cbar2.formatter.set_powerlimits((0, 0))
    ax[2].set_title(r"$d_{\theta}$")

    plt.savefig(figdir["png"]/Path('field_Up_'+str(int(disp*1000)) + 'by1000'+'.png'), transparent=True, bbox_inches='tight', dpi=dpi)
    plt.savefig(figdir["pdf"]/Path('field_Up_'+str(int(disp*1000)) + 'by1000'+'.pdf'), transparent=True, bbox_inches='tight', dpi=dpi)


    # Stress plot
    x = input_el[:, 0]
    y = input_el[:, 1]
    triang = tri.Triangulation(x, y)
    triAnalyzer = tri.TriAnalyzer(triang)
    mask = triAnalyzer.get_flat_tri_mask(min_circle_ratio=0.1, rescale=False)
    triang.set_mask(mask)


    fig, ax = plt.subplots(figsize=(9.5, 2), ncols=3)
    ax[0].set_aspect('equal')
    tpc0 = ax[0].tripcolor(triang, stress_11, shading='gouraud', rasterized=True)
    cbar0 = fig.colorbar(tpc0, ax = ax[0])
    cbar0.formatter.set_powerlimits((0, 0))
    ax[0].set_title(r"$\sigma_{\theta_{11}}$")

    ax[1].set_aspect('equal')
    tpc1 = ax[1].tripcolor(triang, stress_22, shading='gouraud', rasterized=True)
    cbar1 = fig.colorbar(tpc1, ax = ax[1])
    cbar1.formatter.set_powerlimits((0, 0))
    ax[1].set_title(r"$\sigma_{\theta_{22}}$")

    ax[2].set_aspect('equal')
    tpc2 = ax[2].tripcolor(triang, stress_12, shading='gouraud', rasterized=True)
    cbar2 = fig.colorbar(tpc2, ax = ax[2])
    cbar2.formatter.set_powerlimits((0, 0))
    ax[2].set_title(r"$\sigma_{\theta_{12}}$")

    plt.savefig(figdir["png"]/Path('stress_Up_'+str(int(disp*1000)) + 'by1000'+'.png'), transparent=True, bbox_inches='tight', dpi=dpi)
    plt.savefig(figdir["pdf"]/Path('stress_Up_'+str(int(disp*1000)) + 'by1000'+'.pdf'), transparent=True, bbox_inches='tight', dpi=dpi)


    # Principal stress plot
    fig, ax = plt.subplots(figsize=(6, 2), ncols=2)
    ax[0].set_aspect('equal')
    tpc0 = ax[0].tripcolor(triang, stress_1, shading='gouraud', rasterized=True)
    cbar0 = fig.colorbar(tpc0, ax = ax[0])
    cbar0.formatter.set_powerlimits((0, 0))
    ax[0].set_title(r"$\sigma_{\theta_1}$")

    ax[1].set_aspect('equal')
    tpc1 = ax[1].tripcolor(triang, stress_2, shading='gouraud', rasterized=True)
    cbar1 = fig.colorbar(tpc1, ax = ax[1])
    cbar1.formatter.set_powerlimits((0, 0))
    ax[1].set_title(r"$\sigma_{\theta_2}$")

    plt.savefig(figdir["png"]/Path('stress_p_Up_'+str(int(disp*1000)) + 'by1000'+'.png'), transparent=True, bbox_inches='tight', dpi=dpi)
    plt.savefig(figdir["pdf"]/Path('stress_p_Up_'+str(int(disp*1000)) + 'by1000'+'.pdf'), transparent=True, bbox_inches='tight', dpi=dpi)
