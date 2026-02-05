from typing import Tuple, List
import numpy as np
import pytraj as pt
from pytraj.utils.get_common_objects import get_data_from_dtype
from pytraj.datasets.datasetlist import DatasetList
from pytraj import iterframe_master

def get_solvent_donor_hbond_metrics(
    topfile,
    traj_path,
    hbond_accept_idx, #use pytraj index
    solvent_o_type,
    solvent_h_offset: List[int],
    min_ho_angle_deg: float = 150.0,
    min_ho_distance: float = 3,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Compute per-frame solvent-solute hydrogen-bond: solvent is the hydrogen bond donor and solute is the hydrogen bond acceptor.

    Parameters
    ----------
    topfile
        Path to the AMBER topology file (.prmtop).
    traj_path
        Path to a trajectory file.
    hbond_accept_idx
        1-based atom index of the hydrogen bond acceptor (on the solute)
    solvent_o_type
        Topology atom `type` used to identify solvent hydrogen bond donor.
    solvent_h_offset
        A list of offsets from solute hydrogen bond donor index to the hydrogen index used for the angle.
        (to deal with scenario like water where there are two hydrogen bond donor hydrogens)
    min_ho_angle_deg
        Minimum H–O_w···O_carbonyl angle (degrees) to count as H-bonded.
    min_ho_distance
        Minimum O_w···O_carbonyl distances to count as H-bonded.

    Returns
    -------
    mean_ow_oc_distances
        List of length `n_frames`: mean O_w–O_carbonyl distance for qualifying waters
        in each frame (or `fill_value` if none).
    mean_ho_oc_angles
        List of length `n_frames`: mean H–O_w···O_carbonyl angle for qualifying waters
        in each frame (or `fill_value` if none).
    hbond_counts
        List of length `n_frames`: number of qualifying waters per frame.
    """
    
    topology = pt.load_topology(topfile)
    parm = topology.simplify()
    neighbors = DatasetList()   
    traj = pt.iterload(traj_path, topfile)                                                                                   
    
    #MZ: This part of the code is adapted from https://github.com/Amber-MD/pytraj/blob/33b4ff73b3a2faacd4c633c38965de2ed4d59e44/pytraj/actions/utilities.py#L359
    #MZ: I don't know why pt.search_neighbors doesn't work in the current version of pytraj
    for idx, frame in enumerate(iterframe_master(traj)):
        topology.set_reference(frame)
        selected_indices = topology.select(f'(@{hbond_accept_idx}<@{min_ho_distance})|(@{hbond_accept_idx}==@{min_ho_distance})')
        neighbors.append({str(idx): np.asarray(selected_indices)})
    neighbors = get_data_from_dtype(neighbors, 'dataset')
    
    n_frames = len(neighbors)
    mean_dists: List[float] = [0] * n_frames
    mean_angles: List[float] = [0] * n_frames
    counts: List[int] = [0] * n_frames

    o_carb_sel = f"@{hbond_accept_idx}"

    for f, neigh in enumerate(neighbors):
        frame = traj[f : f + 1]
        dists_f: List[float] = []
        angles_f: List[float] = []

        for i in neigh:
            if parm.atoms[i].type != solvent_o_type: 
                continue

            o_w_sel = f"@{i + 1}"
            # Angle: H_w - O_w - O_carbonyl
            best_angle = -np.inf
            for off in solvent_h_offset:
                h_sel = f"@{i + off + 1}"
                ang = pt.angle(frame, f"{o_w_sel} {h_sel} {o_carb_sel}")[0]
                if ang > best_angle:
                    best_angle = ang
            if best_angle >= min_ho_angle_deg:
                ow_oc_dist = pt.distance(frame, f"{o_w_sel} {o_carb_sel}")[0]
                dists_f.append(ow_oc_dist)
                angles_f.append(best_angle)

        counts[f] = len(dists_f)
        if dists_f:
            mean_dists[f] = float(np.mean(dists_f))
            mean_angles[f] = float(np.mean(angles_f))
    return mean_dists, mean_angles, counts

