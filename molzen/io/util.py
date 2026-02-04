import numpy as np
from typing import List, Optional

from molzen.amino_acids import aa2long, aa2num, oneletter_code, ncaas, num2aa, aa_1_to_3
from molzen.ptable import ALL_SYMBOLS
from pathlib import Path

def parse_amber_rst7(path_rst7) -> List[List[float]]:
    """
    Parse an ASCII AMBER restart (.rst7) file.
    
    Args:
        path_rst7 (str): Path to the .rst7 file.
        
    Notes
    -----
    Format (common ASCII rst7):
    line 1: title/comment
    line 2: NATOM [time]
    then: 3*NATOM floats for coordinates (x,y,z per atom)

    Returns
    -------
    coords : np.ndarray
        Shape (N, 3). Coordinates in Angstrom.

    """
    path = Path(path_rst7)
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: too few lines to be a valid rst7.")

    header = lines[1].split()
    if not header:
        raise ValueError(f"{path}: cannot parse NATOM from line 2.")
    natom = int(header[0])

    # Flatten all numeric tokens after line 2 (coords + optional vels + optional box)
    tokens: List[str] = []
    for line in lines[2:]:
        tokens.extend(line.split())

    ncoord = 3 * natom
    if len(tokens) < ncoord:
        raise ValueError(
            f"{path}: not enough numeric values for coordinates: "
            f"got {len(tokens)}, need at least {ncoord}."
        )

    coord_vals = list(map(float, tokens[:ncoord]))
    coords = [coord_vals[i:i + 3] for i in range(0, ncoord, 3)]

    return np.array(coords)

def write_amber_rst7(
    coords: np.ndarray,
    out_path: str,
    title: str = "Created by write_amber_rst7"
) -> None:
    """
    Write an ASCII AMBER restart (rst7) file from an (N, 3) coordinate array.

    Parameters
    ----------
    coords : np.ndarray
        Shape (N, 3). Coordinates in Angstrom.
    out_path : str
        Output .rst7 file path.
    title : str
        First-line title.

    Notes
    -----
    - Writes coordinates only (no velocities, no box).
    - Uses 12.7f formatting and 6 floats per line (two atoms per line).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3); got {coords.shape}")

    natom = coords.shape[0]
    out_path = Path(out_path)

    flat = coords.reshape(-1)

    with out_path.open("w") as f:
        f.write(f"{title}\n")
        f.write(f"{natom:5d}\n")

        for i in range(0, len(flat), 6):
            chunk = flat[i:i + 6]
            f.write("".join(f"{v:12.7f}" for v in chunk) + "\n")
            
def parse_xyz(xyz_fp):
    """Parse a .xyz with potentially multiple frames in it.

    Args:
        xyz_fp (str): Path to the .xyz file.
    """

    xyzs, elements, comments = [], [], []

    with open(xyz_fp, "r") as f:
        lines = f.readlines()

        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # read number of atoms
            natoms = int(line)
            i += 1

            # read comment line
            comment = lines[i].strip()
            i += 1

            # read atom positions
            xyz = []
            for j in range(natoms):
                if i >= len(lines):
                    break

                split = lines[i].strip().split()
                if len(split) != 4:
                    break  # end of molecule or file

                element = split[0]
                x = float(split[1])
                y = float(split[2])
                z = float(split[3])
                xyz.append([x, y, z])

                # not my favorite way to handle this.
                if len(elements) < natoms:
                    elements.append(element)

                i += 1

            xyzs.append(xyz)
            comments.append(comment)

    return dict(xyz=np.array(xyzs), elements=elements, comments=comments)


def write_xyz(
    filename: str,
    xyz: np.ndarray,
    symbols: list[str],
    comments: Optional[list[str]] = None,
    return_str: bool = False,
):
    """Write XYZ file or return the string representation of the XYZ file.

    Args:
        filename: path  to the output file.
        xyz: coordinates of atoms -- shape (B, N, 3) or (N, 3)
        symbols: list of elements
        comments: list of comments for each frame (optional, default is empty strings)
        return_str: if True, return the string representation instead of writing to a file.
    """
    assert xyz.ndim in (2, 3)

    if xyz.ndim == 2:
        xyz = xyz[None, ...]  # add batch dimension
    B, N, _ = xyz.shape
    assert len(symbols) == N, "Number of symbols must match number of atoms."

    if comments is None:
        comments = ["##"] * B

    outstr = ""

    for b in range(B):
        outstr += f"{N}\n"
        outstr += f"{comments[b]}\n"
        for n in range(N):
            outstr += f"{symbols[n]} {xyz[b, n, 0]} {xyz[b, n, 1]} {xyz[b, n, 2]}\n"

    if return_str:
        return outstr

    with open(filename, "w") as f:
        f.write(outstr)


MOL2_HETATM_DTYPES = np.dtype(
    [
        ("atom_idx", "i4"),
        ("atom_name", "U8"),
        ("atom_type", "U8"),
        ("element", "U4"),
        ("res_name", "U3"),
        ("xyz", "f4", (3,)),
    ]
)


_ELEMENT_SYMBOLS = set(ALL_SYMBOLS)


def _infer_element_from_atom_name(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2:
        cand2 = letters[:2].capitalize()
        if cand2 in _ELEMENT_SYMBOLS:
            return cand2
    cand1 = letters[0].upper()
    if cand1 in _ELEMENT_SYMBOLS:
        return cand1
    return ""


def parse_mol2(mol2_fp: str):
    """Parse a .mol2 file to extract atom information.

    Args:
        mol2_fp (str): Path to the .mol2 file.
    """
    hetatm_data = []

    with open(mol2_fp, "r") as f:
        lines = f.readlines()

        atom_section = False

        for line in lines:
            line = line.strip()
            if line.startswith("@<TRIPOS>ATOM"):
                atom_section = True
                continue
            elif line.startswith("@<TRIPOS>"):
                atom_section = False

            if atom_section:
                split = line.split()
                if len(split) < 6:
                    continue  # malformed line

                atom_idx = int(split[0])
                atom_name = split[1]
                x = float(split[2])
                y = float(split[3])
                z = float(split[4])
                atom_type = split[5]
                element = _infer_element_from_atom_name(atom_name)
                if not element:
                    element = atom_type.split(".")[0].capitalize()
                res_name = split[7] if len(split) > 7 else ""

                het = np.array(
                    (
                        atom_idx,
                        atom_name,
                        atom_type,
                        element,
                        res_name,
                        np.array([x, y, z]),
                    ),
                    dtype=MOL2_HETATM_DTYPES,
                )
                hetatm_data.append(het)

    return dict(hetatm=np.array(hetatm_data))


HETATM_DTYPES = np.dtype(
    [
        ("atom_name", "U4"),
        ("res_name", "U3"),
        ("chain_id", "U1"),
        ("res_num", "i4"),
        ("xyz", "f4", (3,)),
    ]
)


def parse_pdb(pdb_fp: str):
    """Parse a .pdb file to extract atom information. Returns residue-level data for amino acids, atom-level data for HETATM entries.

    Args:
        pdb_fp (str): Path to the .pdb file.
    """
    aa_atom_idx = [
        {name.strip(): i for i, name in enumerate(long) if name is not None}
        for long in aa2long
    ]

    xyz = []
    seq = []
    hetatm_data = []

    with open(pdb_fp, "r") as f:
        lines = f.readlines()

    cur_res_id = None
    cur_xyz = None

    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue

        if line.startswith("ATOM"):
            atom_name = line[12:16].strip()

            res_name = line[17:20].strip()
            chain_id = line[21]
            resnum = line[22:26].strip()

            res_id = (chain_id, resnum)

            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            if res_id != cur_res_id:
                if cur_res_id is not None:
                    # save previous residue
                    xyz.append(cur_xyz)

                # this will raise a KeyError for non-standard residues
                res_oneletter = oneletter_code.get(res_name, None)
                if res_oneletter is None:
                    res_oneletter = ncaas.get(res_name, None)["canonical_one_letter"]
                    if res_oneletter is None:
                        raise KeyError(f"Unknown residue name: {res_name}")
                seq.append(res_oneletter)
                cur_res_id = res_id
                cur_xyz = np.full((27, 3), np.nan)

            aa_index = aa2num[res_name]
            atom_map = aa_atom_idx[aa_index]

            if atom_name in atom_map:
                atom_idx = atom_map[atom_name]
                cur_xyz[atom_idx] = np.array([x, y, z])

        elif line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21]
            resnum = line[22:26].strip()
            res_id = (chain_id, resnum)

            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            het = np.array(
                (atom_name, res_name, chain_id, int(resnum), np.array([x, y, z])),
                dtype=HETATM_DTYPES,
            )
            hetatm_data.append(het)

    # save last residue
    if cur_res_id is not None:
        xyz.append(cur_xyz)

    return dict(xyz=np.array(xyz), seq="".join(seq), hetatm=np.array(hetatm_data))


def write_pdb(file_path, xyz, seq, chains=None, hetatm=None):
    """Write a PDB file from residue-level xyz data.

    Args:
        file_path: Path to output PDB file.
        xyz: Array of atomic coords (Nres, 27|14, 3)
        seq: Amino acid sequence integers or one-letter codes (Nres,)
        chains: Optional list of chain IDs for each residue. If None, all residues will be in chain 'A'.
        hetatm: Optional array of HETATM entries with dtype HETATM_DTYPES
    """
    aa_atom_idx = [
        {name.strip(): i for i, name in enumerate(long) if name is not None}
        for long in aa2long
    ]

    with open(file_path, "w") as f:
        atom_counter = 1
        res_counter = 1

        Nres = xyz.shape[0]

        if chains is None:
            chains = ["A"] * Nres

        for i in range(Nres):
            res_name = aa_1_to_3[seq[i]] if isinstance(seq[i], str) else num2aa[seq[i]]
            chain_id = chains[i]
            atom_map = aa_atom_idx[aa2num[res_name]]

            for atom_name, atom_idx in atom_map.items():
                coord = xyz[i, atom_idx]
                if np.any(np.isnan(coord)):
                    print(
                        f"Warning: missing coordinate for residue {res_counter} atom {atom_name}, skipping."
                    )
                    continue  # skip missing atoms

                f.write(
                    f"ATOM  {atom_counter:5d} {atom_name:>4} {res_name:>3} {chain_id}{res_counter:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           {atom_name[0]:>2}\n"
                )
                atom_counter += 1

            res_counter += 1

        if hetatm is not None:
            for het in hetatm:
                atom_name = het["atom_name"]
                res_name = het["res_name"]
                chain_id = het["chain_id"]
                res_num = het["res_num"]
                coord = het["xyz"]

                f.write(
                    f"HETATM{atom_counter:5d} {atom_name:>4} {res_name:>3} {chain_id}{res_num:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           {atom_name[0]:>2}\n"
                )
                atom_counter += 1

        f.write("END\n")
