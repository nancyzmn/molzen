import pytraj as pt
import numpy as np

def get_bending_angles(
    traj_path: str,
    topfile: str,
    mask1: str,
    mask2: str,
    mask3: str
) -> np.ndarray:
    """
    Compute a signed bending angle (in degrees) defined by midpoints of three bonds.
    (AMBER mask example: :JF4@C12,C11)

    It then computes the angle between vectors (p1 - p2) and (p3 - p2). The sign is
    determined using a reference direction given by the bond vector between the two
    atoms in `mask2`: if dot(cross(v1, v2), ref) < 0, the angle is negated. 
    Returned angles are in degrees in the range [-180, 180].

    Parameters
    ----------
    traj_path
        Path to a trajectory file.
    topfile
        Path to the AMBER topology file (.prmtop).
    mask1, mask2, mask3
        Pytraj/Amber selection masks. Each mask must select exactly two atoms
        (used to form a midpoint). 

    Returns
    -------
    numpy.ndarray
        Array of shape (n_frames,) with signed angles in degrees.

    """
    top = pt.load_topology(topfile)
    traj = pt.iterload([traj_path], topfile)
    traj.top.set_reference(traj[0])
    idx1 = top.atom_indices(mask1)
    idx2 = top.atom_indices(mask2)
    idx3 = top.atom_indices(mask3)

    for name, idx in [("mask1", idx1), ("mask2", idx2), ("mask3", idx3)]:
        if len(idx) != 2:
            raise ValueError(f"{name} must select exactly 2 atoms; got {len(idx)} from {name}.")

    def _midpoint(frame, i, j) -> np.ndarray:
        return 0.5 * (np.asarray(frame.atom(i), dtype=float) + np.asarray(frame.atom(j), dtype=float))

    def _unit(vec: np.ndarray, label: str) -> np.ndarray:
        n = np.linalg.norm(vec)
        if n < 1e-12:
            raise ValueError(f"Zero-length (or near-zero) vector encountered for {label}.")
        return vec / n

    angles_deg = np.empty(len(traj), dtype=float)

    for k, frame in enumerate(traj):
        p1 = _midpoint(frame, idx1[0], idx1[1])
        p2 = _midpoint(frame, idx2[0], idx2[1])
        p3 = _midpoint(frame, idx3[0], idx3[1])

        v1 = _unit(p1 - p2, "v1=(p1-p2)")
        v2 = _unit(p3 - p2, "v2=(p3-p2)")

        cross = np.cross(v1, v2)
        dot = float(np.dot(v1, v2))
        angle = float(np.arctan2(np.linalg.norm(cross), dot))

        # Sign via reference bond direction from mask2 atoms (ordered)
        ref = _unit(np.asarray(frame.atom(idx2[0])) - np.asarray(frame.atom(idx2[1])), "ref=(mask2 atom0 - atom1)")
        if np.dot(cross, ref) < 0.0:
            angle = -angle

        angles_deg[k] = np.degrees(angle)

    return angles_deg
