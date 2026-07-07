#!/usr/bin/env python3
"""TDDFT/TDA Dyson orbitals from Q-Chem, Gaussian 16, and ORCA files.

The determinant-overlap expression follows the Dyson-orbital supporting
information in this directory.  For two separately optimized SCF references,
spin-orbital overlaps are built as

    <chi_p^cation | chi_q^neutral> = C_p(cation)^T S_AO C_q(neutral)

where S_AO is read from the checkpoint when available or reconstructed from a
complete MO set when the producing program does not write it explicitly.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from math import pi
from pathlib import Path

import numpy as np


HARTREE_TO_EV = 27.211386245988
BOHR_PER_ANGSTROM = 1.8897259886


ELEMENT_Z = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "BR": 35,
    "I": 53,
}

Z_ELEMENT = {z: symbol.capitalize() for symbol, z in ELEMENT_Z.items()}


@dataclass
class Shell:
    atom_index: int
    shell_type: int
    exponents: list[float]
    coefficients: list[float]
    p_coefficients: list[float] | None = None


@dataclass
class OrbitalData:
    path: Path
    nbf: int
    n_alpha: int
    n_beta: int
    overlap: np.ndarray
    alpha_mo: np.ndarray
    beta_mo: np.ndarray
    alpha_energies: np.ndarray
    beta_energies: np.ndarray
    lines: list[str] | None = None
    charge: int | None = None
    multiplicity: int | None = None
    atomic_numbers: np.ndarray | None = None
    coordinates_bohr: np.ndarray | None = None
    shells: list[Shell] = field(default_factory=list)
    molden_normalized_basis: bool = False


@dataclass
class Transition:
    source: int
    target: int
    amplitude: float
    spin: str
    label: str = ""


@dataclass
class TDAState:
    state: int
    excitation_ev: float
    transitions: list[Transition]


@dataclass
class DysonState:
    state: int
    excitation_ev: float
    vertical_ip_ev: float
    mo_coefficients: np.ndarray
    alpha_ao: np.ndarray
    beta_ao: np.ndarray
    total_norm: float
    alpha_norm: float
    beta_norm: float
    dominant_spin: str


@dataclass
class CalculationResult:
    dyson_states: list[DysonState]
    output_fchk: Path | None
    output_report: Path | None
    output_molden: Path | None


def _float_token(token: str) -> float:
    return float(token.replace("D", "E"))


def _field_header(line: str) -> tuple[str, str, str] | None:
    if not line or line[0].isspace() or len(line) < 44:
        return None
    label = line[:43].strip()
    type_code = line[43].strip()
    rest = line[44:].strip()
    if type_code not in {"I", "R", "C", "L"}:
        return None
    return label, type_code, rest


def fchk_scalar(lines: list[str], label: str, cast=float):
    for line in lines:
        header = _field_header(line)
        if header is None:
            continue
        found_label, _type_code, rest = header
        if found_label == label:
            return cast(rest.split()[-1])
    raise KeyError(f"Field not found in fchk: {label}")


def fchk_array(lines: list[str], label: str) -> np.ndarray:
    for idx, line in enumerate(lines):
        header = _field_header(line)
        if header is None:
            continue
        found_label, type_code, rest = header
        if found_label != label:
            continue
        match = re.search(r"N=\s*(\d+)", rest)
        if match is None:
            raise ValueError(f"Field {label!r} is not an array")
        n_values = int(match.group(1))
        values: list[float] = []
        cursor = idx + 1
        while len(values) < n_values and cursor < len(lines):
            tokens = lines[cursor].split()
            if type_code == "I":
                values.extend(float(int(token)) for token in tokens)
            else:
                values.extend(_float_token(token) for token in tokens)
            cursor += 1
        if len(values) != n_values:
            raise ValueError(
                f"Field {label!r} expected {n_values} values, read {len(values)}"
            )
        return np.array(values, dtype=float)
    raise KeyError(f"Field not found in fchk: {label}")


def unpack_lower_triangle(values: np.ndarray, n: int) -> np.ndarray:
    expected = n * (n + 1) // 2
    if len(values) != expected:
        raise ValueError(f"Packed matrix has {len(values)} values, expected {expected}")
    matrix = np.zeros((n, n), dtype=float)
    cursor = 0
    for i in range(n):
        for j in range(i + 1):
            matrix[i, j] = values[cursor]
            matrix[j, i] = values[cursor]
            cursor += 1
    return matrix


def pack_lower_triangle(matrix: np.ndarray) -> np.ndarray:
    values = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1):
            values.append(matrix[i, j])
    return np.array(values, dtype=float)


def infer_overlap_from_mos(mo_coeff: np.ndarray) -> np.ndarray:
    """Return S such that C.T @ S @ C = I for a complete MO matrix."""

    return np.linalg.inv(mo_coeff.T) @ np.linalg.inv(mo_coeff)


def read_fchk_shells(lines: list[str]) -> list[Shell]:
    try:
        shell_types = fchk_array(lines, "Shell types").astype(int)
        primitive_counts = fchk_array(lines, "Number of primitives per shell").astype(int)
        shell_atoms = fchk_array(lines, "Shell to atom map").astype(int)
        exponents = fchk_array(lines, "Primitive exponents")
        coefficients = fchk_array(lines, "Contraction coefficients")
    except KeyError:
        return []

    try:
        p_coefficients = fchk_array(lines, "P(S=P) Contraction coefficients")
    except KeyError:
        p_coefficients = None

    shells: list[Shell] = []
    cursor = 0
    for shell_type, n_primitive, atom_index in zip(shell_types, primitive_counts, shell_atoms):
        end = cursor + int(n_primitive)
        p_values = None
        if p_coefficients is not None and int(shell_type) == -1:
            p_values = p_coefficients[cursor:end].tolist()
        shells.append(
            Shell(
                atom_index=int(atom_index),
                shell_type=int(shell_type),
                exponents=exponents[cursor:end].tolist(),
                coefficients=coefficients[cursor:end].tolist(),
                p_coefficients=p_values,
            )
        )
        cursor = end

    return shells


def read_fchk(path: str | Path) -> OrbitalData:
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    nbf = fchk_scalar(lines, "Number of basis functions", int)
    n_alpha = fchk_scalar(lines, "Number of alpha electrons", int)
    n_beta = fchk_scalar(lines, "Number of beta electrons", int)

    try:
        overlap = unpack_lower_triangle(fchk_array(lines, "Overlap Matrix"), nbf)
    except KeyError:
        # Gaussian fchk often omits S_AO but includes X with X.T S X = I.
        x_orth = fchk_array(lines, "Orthonormal basis").reshape((nbf, nbf), order="F")
        overlap = np.linalg.inv(x_orth.T) @ np.linalg.inv(x_orth)

    alpha_raw = fchk_array(lines, "Alpha MO coefficients")
    beta_raw = fchk_array(lines, "Beta MO coefficients")
    if alpha_raw.size != nbf * nbf or beta_raw.size != nbf * nbf:
        raise ValueError("This script expects square MO coefficient blocks in the fchk")

    atomic_numbers = None
    coordinates = None
    try:
        atomic_numbers = fchk_array(lines, "Atomic numbers").astype(int)
        coordinates = fchk_array(lines, "Current cartesian coordinates")
    except KeyError:
        pass

    return OrbitalData(
        path=path,
        nbf=nbf,
        n_alpha=n_alpha,
        n_beta=n_beta,
        overlap=overlap,
        alpha_mo=alpha_raw.reshape((nbf, nbf), order="F"),
        beta_mo=beta_raw.reshape((nbf, nbf), order="F"),
        alpha_energies=fchk_array(lines, "Alpha Orbital Energies"),
        beta_energies=fchk_array(lines, "Beta Orbital Energies"),
        lines=lines,
        charge=_optional_fchk_scalar(lines, "Charge", int),
        multiplicity=_optional_fchk_scalar(lines, "Multiplicity", int),
        atomic_numbers=atomic_numbers,
        coordinates_bohr=coordinates,
        shells=read_fchk_shells(lines),
    )


def _optional_fchk_scalar(lines: list[str], label: str, cast=float):
    try:
        return fchk_scalar(lines, label, cast)
    except KeyError:
        return None


def check_orthonormal(mo: np.ndarray, overlap: np.ndarray, label: str) -> None:
    metric = mo.T @ overlap @ mo
    err = np.linalg.norm(metric - np.eye(metric.shape[0]))
    if err > 1.0e-5:
        raise ValueError(f"{label} MOs are not S-orthonormal; ||C^TSC-I||={err:.3e}")


def run_converter(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        )


def ensure_gaussian_fchk(
    checkpoint: str | Path | None,
    fchk_path: str | Path | None,
    formchk_command: str,
) -> Path:
    if fchk_path is not None:
        fchk = Path(fchk_path)
        if fchk.exists():
            return fchk
    if checkpoint is None:
        raise ValueError("Gaussian input needs either --*-fchk or --*-checkpoint")
    chk = Path(checkpoint)
    fchk = Path(fchk_path) if fchk_path is not None else chk.with_suffix(".fchk")
    if not fchk.exists():
        run_converter([formchk_command, chk.name, fchk.name], chk.parent)
    return fchk


def ensure_orca_molden(
    gbw_path: str | Path | None,
    molden_path: str | Path | None,
    orca_2mkl_command: str,
) -> Path:
    if molden_path is not None:
        molden = Path(molden_path)
        if molden.exists():
            return molden
    if gbw_path is None:
        raise ValueError("ORCA input needs either --*-molden or --*-gbw")
    gbw = Path(gbw_path)
    molden = Path(molden_path) if molden_path is not None else gbw.with_suffix(".molden.input")
    if not molden.exists():
        run_converter([orca_2mkl_command, gbw.with_suffix("").name, "-molden"], gbw.parent)
    return molden


def parse_scf_energy(output_lines: list[str], program: str) -> float:
    if program == "qchem":
        for line in output_lines:
            match = re.search(r"SCF\s+energy\s*=\s*([+-]?\d+\.\d+)", line)
            if match:
                return float(match.group(1))
    elif program == "gaussian":
        for line in output_lines:
            match = re.search(r"SCF Done:\s+E\(.+?\)\s*=\s*([+-]?\d+\.\d+)", line)
            if match:
                return float(match.group(1))
    elif program == "orca":
        in_scf_energy = False
        for line in output_lines:
            if "TOTAL SCF ENERGY" in line:
                in_scf_energy = True
                continue
            if in_scf_energy:
                match = re.search(r"Total Energy\s*:\s*([+-]?\d+\.\d+)", line)
                if match:
                    return float(match.group(1))
    raise ValueError(f"Could not find {program} SCF energy in output file")


def parse_qchem_tda_states(output_lines: list[str], cation: OrbitalData) -> list[TDAState]:
    states: list[TDAState] = []
    current: TDAState | None = None
    in_canonical_block = False

    state_re = re.compile(
        r"^\s*Excited state\s+(\d+):\s+([+-]?\d+(?:\.\d+)?)\s+eV"
    )
    trans_re = re.compile(
        r"([DSV])\(\s*(\d+)\)\s*-->\s*([DSV])\(\s*(\d+)\)\s+"
        r"ampl(?:itude)?\s*=\s*([+-]?(?:\d+\.\d*|\.\d+)(?:[Ee][+-]?\d+)?)"
        r"\s+(alpha|beta)",
        re.IGNORECASE,
    )

    for line in output_lines:
        if "TDA excitation amplitudes in the canonical MO basis" in line:
            in_canonical_block = True
            current = None
            continue
        if not in_canonical_block:
            continue
        if "Mulliken & Loewdin analysis" in line:
            break

        state_match = state_re.search(line)
        if state_match:
            current = TDAState(
                state=int(state_match.group(1)),
                excitation_ev=float(state_match.group(2)),
                transitions=[],
            )
            states.append(current)
            continue

        trans_match = trans_re.search(line)
        if trans_match and current is not None:
            spin = trans_match.group(6).lower()
            source = map_qchem_open_shell_orbital(
                trans_match.group(1),
                int(trans_match.group(2)),
                spin,
                cation.nbf,
                cation.n_alpha,
                cation.n_beta,
            )
            target = map_qchem_open_shell_orbital(
                trans_match.group(3),
                int(trans_match.group(4)),
                spin,
                cation.nbf,
                cation.n_alpha,
                cation.n_beta,
            )
            current.transitions.append(
                Transition(
                    source=source,
                    target=target,
                    amplitude=float(trans_match.group(5)),
                    spin=spin,
                    label=line.strip(),
                )
            )

    if not states:
        raise ValueError("No Q-Chem canonical TDA excitation amplitudes were found")
    return states


def parse_gaussian_tda_states(output_lines: list[str], cation: OrbitalData) -> list[TDAState]:
    states: list[TDAState] = []
    current: TDAState | None = None
    state_re = re.compile(r"Excited State\s+(\d+):.*?([+-]?\d+(?:\.\d+)?)\s+eV")
    trans_re = re.compile(
        r"^\s*(\d+)([AB])\s*->\s*(\d+)([AB])\s+"
        r"([+-]?(?:\d+\.\d*|\.\d+)(?:[Ee][+-]?\d+)?)"
    )

    for line in output_lines:
        state_match = state_re.search(line)
        if state_match:
            current = TDAState(
                state=int(state_match.group(1)),
                excitation_ev=float(state_match.group(2)),
                transitions=[],
            )
            states.append(current)
            continue

        trans_match = trans_re.search(line)
        if trans_match and current is not None:
            from_spin = trans_match.group(2)
            to_spin = trans_match.group(4)
            if from_spin != to_spin:
                raise ValueError(f"Spin-changing Gaussian transition is unsupported: {line}")
            spin = "alpha" if from_spin == "A" else "beta"
            offset = 0 if spin == "alpha" else cation.nbf
            current.transitions.append(
                Transition(
                    source=offset + int(trans_match.group(1)) - 1,
                    target=offset + int(trans_match.group(3)) - 1,
                    amplitude=float(trans_match.group(5)),
                    spin=spin,
                    label=line.strip(),
                )
            )

    if not states:
        raise ValueError("No Gaussian TDDFT/TDA excited-state amplitudes were found")
    return states


def parse_orca_tda_states(output_lines: list[str], cation: OrbitalData) -> list[TDAState]:
    states: list[TDAState] = []
    current: TDAState | None = None
    in_block = False
    state_re = re.compile(
        r"^\s*STATE\s+(\d+):\s+E=\s+[+-]?\d+\.\d+\s+au\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+eV"
    )
    trans_re = re.compile(
        r"^\s*(\d+)([ab])\s*->\s*(\d+)([ab])\s*:"
        r".*?\(c=\s*([+-]?(?:\d+\.\d*|\.\d+)(?:[Ee][+-]?\d+)?)\)"
    )

    for line in output_lines:
        if "TD-DFT/TDA EXCITED STATES" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if "Storing amplitudes" in line or "TD-DFT/TDA-EXCITATION SPECTRA" in line:
            break

        state_match = state_re.search(line)
        if state_match:
            current = TDAState(
                state=int(state_match.group(1)),
                excitation_ev=float(state_match.group(2)),
                transitions=[],
            )
            states.append(current)
            continue

        trans_match = trans_re.search(line)
        if trans_match and current is not None:
            from_spin = trans_match.group(2)
            to_spin = trans_match.group(4)
            if from_spin != to_spin:
                raise ValueError(f"Spin-changing ORCA transition is unsupported: {line}")
            spin = "alpha" if from_spin == "a" else "beta"
            offset = 0 if spin == "alpha" else cation.nbf
            current.transitions.append(
                Transition(
                    source=offset + int(trans_match.group(1)),
                    target=offset + int(trans_match.group(3)),
                    amplitude=float(trans_match.group(5)),
                    spin=spin,
                    label=line.strip(),
                )
            )

    if not states:
        raise ValueError("No ORCA TDDFT/TDA excited-state amplitudes were found")
    return states


def map_qchem_open_shell_orbital(
    orbital_type: str,
    orbital_number: int,
    spin: str,
    n_spatial_mos: int,
    n_alpha_occ: int,
    n_beta_occ: int,
) -> int:
    if orbital_number < 1:
        raise ValueError("Q-Chem orbital numbers are one-based")

    spin = spin.lower()
    orbital_type = orbital_type.upper()
    n_doubly = min(n_alpha_occ, n_beta_occ)
    n_singly = abs(n_alpha_occ - n_beta_occ)
    alpha_has_somo = n_alpha_occ > n_beta_occ
    beta_has_somo = n_beta_occ > n_alpha_occ

    if spin == "alpha":
        offset = 0
        n_occ = n_alpha_occ
        spin_has_somo = alpha_has_somo
    elif spin == "beta":
        offset = n_spatial_mos
        n_occ = n_beta_occ
        spin_has_somo = beta_has_somo
    else:
        raise ValueError(f"Unknown spin label: {spin}")

    if orbital_type == "D":
        if orbital_number > n_doubly:
            raise ValueError(f"D({orbital_number}) exceeds {n_doubly} doubly occupied MOs")
        return offset + orbital_number - 1
    if orbital_type == "S":
        if orbital_number > n_singly:
            raise ValueError(f"S({orbital_number}) exceeds {n_singly} singly occupied MOs")
        if spin_has_somo:
            return offset + n_doubly + orbital_number - 1
        return offset + n_occ + orbital_number - 1
    if orbital_type == "V":
        if spin_has_somo:
            return offset + n_occ + orbital_number - 1
        return offset + n_occ + n_singly + orbital_number - 1
    raise ValueError(f"Unknown orbital type: {orbital_type}")


def parse_orca_charge_multiplicity(output_lines: list[str]) -> tuple[int, int]:
    charge = None
    multiplicity = None
    for line in output_lines:
        match_charge = re.search(r"Total Charge\s+Charge\s+\.\.\.\.\s+(-?\d+)", line)
        if match_charge:
            charge = int(match_charge.group(1))
        match_mult = re.search(r"Multiplicity\s+Mult\s+\.\.\.\.\s+(\d+)", line)
        if match_mult:
            multiplicity = int(match_mult.group(1))
        if charge is not None and multiplicity is not None:
            return charge, multiplicity
    raise ValueError("Could not find ORCA charge/multiplicity")


def parse_molden(path: str | Path, charge: int, multiplicity: int) -> OrbitalData:
    path = Path(path)
    lines = path.read_text().splitlines()
    atoms: list[tuple[int, np.ndarray]] = []
    shells: list[Shell] = []
    in_atoms = False
    atoms_unit = "AU"
    in_gto = False
    in_mo = False
    current_atom = None
    mo_entries: list[tuple[str, float, float, np.ndarray]] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        upper = line.upper()
        if upper.startswith("[ATOMS]"):
            in_atoms = True
            in_gto = False
            in_mo = False
            atoms_unit = "AU" if "AU" in upper else "ANGS"
            i += 1
            continue
        if upper.startswith("[GTO]"):
            in_atoms = False
            in_gto = True
            in_mo = False
            i += 1
            continue
        if upper.startswith("[MO]"):
            in_atoms = False
            in_gto = False
            in_mo = True
            i += 1
            continue
        if line.startswith("[") and not upper.startswith("[MO]"):
            in_atoms = False
            in_gto = False

        if in_atoms and line:
            tokens = line.split()
            symbol = re.sub(r"\d+$", "", tokens[0]).upper()
            z = int(tokens[2]) if tokens[2].lstrip("-").isdigit() else ELEMENT_Z[symbol]
            coords = np.array([float(tokens[3]), float(tokens[4]), float(tokens[5])])
            if atoms_unit.startswith("ANG"):
                coords *= BOHR_PER_ANGSTROM
            atoms.append((z, coords))

        elif in_gto and line:
            tokens = line.split()
            if len(tokens) >= 2 and tokens[0].isdigit():
                current_atom = int(tokens[0])
            elif tokens[0].lower() in {"s", "p", "d", "f", "g", "sp"}:
                if current_atom is None:
                    raise ValueError("Molden GTO shell found before atom index")
                shell_letter = tokens[0].lower()
                nprim = int(tokens[1])
                exponents = []
                coefficients = []
                p_coefficients = None
                if shell_letter == "sp":
                    p_coefficients = []
                for offset in range(1, nprim + 1):
                    primitive = lines[i + offset].split()
                    exponents.append(float(primitive[0]))
                    coefficients.append(float(primitive[1]))
                    if p_coefficients is not None:
                        if len(primitive) < 3:
                            raise ValueError("Molden sp shell is missing p coefficients")
                        p_coefficients.append(float(primitive[2]))
                shell_type = {"s": 0, "p": 1, "d": -2, "f": -3, "g": -4, "sp": -1}[shell_letter]
                shells.append(Shell(current_atom, shell_type, exponents, coefficients, p_coefficients))
                i += nprim

        elif in_mo and line.startswith("Sym="):
            energy = None
            spin = None
            occ = None
            coeffs = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("Sym="):
                mo_line = lines[i].strip()
                if mo_line.startswith("Ene="):
                    energy = float(mo_line.split("=", 1)[1])
                elif mo_line.startswith("Spin="):
                    spin = mo_line.split("=", 1)[1].strip().lower()
                elif mo_line.startswith("Occup="):
                    occ = float(mo_line.split("=", 1)[1])
                elif mo_line and mo_line.split()[0].isdigit():
                    coeffs.append(float(mo_line.split()[1]))
                i += 1
            if energy is None or spin is None or occ is None:
                raise ValueError(f"Malformed Molden MO block in {path}")
            mo_entries.append((spin, energy, occ, np.array(coeffs, dtype=float)))
            continue
        i += 1

    alpha_entries = [entry for entry in mo_entries if entry[0] == "alpha"]
    beta_entries = [entry for entry in mo_entries if entry[0] == "beta"]
    if not alpha_entries or not beta_entries:
        raise ValueError(f"Could not find both alpha and beta MOs in {path}")

    alpha_mo = np.column_stack([entry[3] for entry in alpha_entries])
    beta_mo = np.column_stack([entry[3] for entry in beta_entries])
    nbf = alpha_mo.shape[0]
    if alpha_mo.shape != (nbf, nbf) or beta_mo.shape != (nbf, nbf):
        raise ValueError("This script expects complete square MO blocks in Molden files")

    return OrbitalData(
        path=path,
        nbf=nbf,
        n_alpha=int(round(sum(entry[2] for entry in alpha_entries))),
        n_beta=int(round(sum(entry[2] for entry in beta_entries))),
        overlap=infer_overlap_from_mos(alpha_mo),
        alpha_mo=alpha_mo,
        beta_mo=beta_mo,
        alpha_energies=np.array([entry[1] for entry in alpha_entries]),
        beta_energies=np.array([entry[1] for entry in beta_entries]),
        charge=charge,
        multiplicity=multiplicity,
        atomic_numbers=np.array([atom[0] for atom in atoms], dtype=int),
        coordinates_bohr=np.concatenate([atom[1] for atom in atoms]),
        shells=shells,
        molden_normalized_basis=True,
    )


def attach_minimal_fchk_lines(data: OrbitalData, title: str) -> None:
    if data.atomic_numbers is None or data.coordinates_bohr is None or not data.shells:
        raise ValueError("Cannot build a minimal fchk without atoms and shells")
    lines: list[str] = [f"{title}\n", "SP        U                             ORCA-Molden\n"]
    natoms = len(data.atomic_numbers)
    n_primitives = sum(len(shell.exponents) for shell in data.shells)
    shell_types = np.array([shell.shell_type for shell in data.shells], dtype=int)
    primitive_counts = np.array([len(shell.exponents) for shell in data.shells], dtype=int)
    shell_atoms = np.array([shell.atom_index for shell in data.shells], dtype=int)
    exponents = np.array([x for shell in data.shells for x in shell.exponents])
    coefficients = np.array([x for shell in data.shells for x in shell.coefficients])
    if any(shell.p_coefficients is not None for shell in data.shells):
        p_coefficients = np.array(
            [
                value
                for shell in data.shells
                for value in (
                    shell.p_coefficients
                    if shell.p_coefficients is not None
                    else [0.0] * len(shell.exponents)
                )
            ]
        )
    else:
        p_coefficients = None
    shell_coords = np.concatenate(
        [data.coordinates_bohr[3 * (shell.atom_index - 1) : 3 * shell.atom_index] for shell in data.shells]
    )

    lines.append(format_fchk_int_scalar("Number of atoms", natoms))
    lines.append(format_fchk_int_scalar("Charge", data.charge or 0))
    lines.append(format_fchk_int_scalar("Multiplicity", data.multiplicity or 1))
    lines.append(format_fchk_int_scalar("Number of electrons", data.n_alpha + data.n_beta))
    lines.append(format_fchk_int_scalar("Number of alpha electrons", data.n_alpha))
    lines.append(format_fchk_int_scalar("Number of beta electrons", data.n_beta))
    lines.extend(format_fchk_int_array("Atomic numbers", data.atomic_numbers))
    lines.extend(format_fchk_array("Current cartesian coordinates", data.coordinates_bohr))
    lines.extend(format_fchk_array("Nuclear charges", data.atomic_numbers.astype(float)))
    lines.append(format_fchk_int_scalar("Number of basis functions", data.nbf))
    lines.append(format_fchk_int_scalar("Number of contracted shells", len(data.shells)))
    lines.append(format_fchk_int_scalar("Number of primitive shells", n_primitives))
    lines.append(format_fchk_int_scalar("Highest angular momentum", int(max(abs(shell_types)))))
    lines.append(format_fchk_int_scalar("Largest degree of contraction", int(max(primitive_counts))))
    lines.extend(format_fchk_int_array("Shell types", shell_types))
    lines.extend(format_fchk_int_array("Number of primitives per shell", primitive_counts))
    lines.extend(format_fchk_int_array("Shell to atom map", shell_atoms))
    lines.extend(format_fchk_array("Primitive exponents", exponents))
    lines.extend(format_fchk_array("Contraction coefficients", coefficients))
    if p_coefficients is not None:
        lines.extend(format_fchk_array("P(S=P) Contraction coefficients", p_coefficients))
    lines.extend(format_fchk_array("Coordinates of each shell", shell_coords))
    lines.extend(format_fchk_array("Overlap Matrix", pack_lower_triangle(data.overlap)))
    lines.append(format_fchk_int_scalar("Number of independent functions", data.nbf))
    lines.extend(format_fchk_array("Alpha MO coefficients", data.alpha_mo.reshape(-1, order="F")))
    lines.extend(format_fchk_array("Beta MO coefficients", data.beta_mo.reshape(-1, order="F")))
    lines.extend(format_fchk_array("Alpha Orbital Energies", data.alpha_energies))
    lines.extend(format_fchk_array("Beta Orbital Energies", data.beta_energies))
    lines.extend(format_fchk_array("Total SCF Density", np.zeros(data.nbf * (data.nbf + 1) // 2)))
    lines.extend(format_fchk_array("Spin SCF Density", np.zeros(data.nbf * (data.nbf + 1) // 2)))
    data.lines = lines


def block_diag(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    zeros_ab = np.zeros((alpha.shape[0], beta.shape[1]), dtype=float)
    zeros_ba = np.zeros((beta.shape[0], alpha.shape[1]), dtype=float)
    return np.block([[alpha, zeros_ab], [zeros_ba, beta]])


def spin_orbital_overlap(neutral: OrbitalData, cation: OrbitalData) -> np.ndarray:
    if neutral.nbf != cation.nbf:
        raise ValueError("Neutral and cation files have different basis sizes")
    if np.linalg.norm(neutral.overlap - cation.overlap) > 1.0e-6:
        raise ValueError("Neutral and cation AO overlap matrices differ")

    check_orthonormal(neutral.alpha_mo, neutral.overlap, "neutral alpha")
    check_orthonormal(neutral.beta_mo, neutral.overlap, "neutral beta")
    check_orthonormal(cation.alpha_mo, neutral.overlap, "cation alpha")
    check_orthonormal(cation.beta_mo, neutral.overlap, "cation beta")

    u_alpha = cation.alpha_mo.T @ neutral.overlap @ neutral.alpha_mo
    u_beta = cation.beta_mo.T @ neutral.overlap @ neutral.beta_mo
    return block_diag(u_alpha, u_beta)


def dyson_ks(cis_coefficients: dict[int, np.ndarray | None], overlap_mo: np.ndarray, occ: dict[int, list[int]]) -> dict[int, np.ndarray]:
    first_excited = next(coeff for coeff in cis_coefficients.values() if coeff is not None)
    n_virtual, n_occupied = first_excited.shape
    n_spin_orbitals = n_occupied + n_virtual
    n_neutral_electrons = n_occupied + 1
    virtual = [idx for idx in range(n_spin_orbitals) if idx not in occ[+1]]

    determinants = np.zeros((n_neutral_electrons, n_virtual, n_occupied), dtype=overlap_mo.dtype)
    ground_determinants = np.zeros(n_neutral_electrons, dtype=overlap_mo.dtype)
    base = np.zeros((n_neutral_electrons, n_neutral_electrons), dtype=overlap_mo.dtype)

    for row, cation_occ in enumerate(occ[+1]):
        for col, neutral_occ in enumerate(occ[0]):
            base[row + 1, col] = overlap_mo[cation_occ, neutral_occ]

    for removed_col in range(n_neutral_electrons):
        base[0, :] = 0.0
        base[0, removed_col] = 1.0
        ground_determinants[removed_col] = np.linalg.det(base)

        for occ_row in range(n_occupied):
            for virt_row, cation_virtual in enumerate(virtual):
                matrix = np.array(base, copy=True)
                for col, neutral_occ in enumerate(occ[0]):
                    matrix[occ_row + 1, col] = overlap_mo[cation_virtual, neutral_occ]
                determinants[removed_col, virt_row, occ_row] = np.linalg.det(matrix)

    dyson: dict[int, np.ndarray] = {}
    for state, coeff in cis_coefficients.items():
        if coeff is None:
            contracted = ground_determinants
        else:
            contracted = np.einsum("bj,ibj->i", coeff, determinants)
        state_dyson = np.zeros(n_spin_orbitals, dtype=overlap_mo.dtype)
        for local_idx, neutral_occ in enumerate(occ[0]):
            state_dyson[neutral_occ] = contracted[local_idx]
        dyson[state] = state_dyson
    return dyson


def build_cis_coefficients(
    states: list[TDAState],
    cation: OrbitalData,
    amplitude_cutoff: float,
) -> tuple[dict[int, np.ndarray | None], dict[int, int]]:
    nmo = cation.nbf
    occ_alpha = list(range(cation.n_alpha))
    occ_beta = list(range(nmo, nmo + cation.n_beta))
    occ_cation = occ_alpha + occ_beta
    virtual_cation = [idx for idx in range(2 * nmo) if idx not in occ_cation]
    coefficients: dict[int, np.ndarray | None] = {0: None}
    kept_counts: dict[int, int] = {}

    for state in states:
        matrix = np.zeros((len(virtual_cation), len(occ_cation)), dtype=float)
        kept = 0
        for transition in state.transitions:
            if abs(transition.amplitude) < amplitude_cutoff:
                continue
            if transition.source not in occ_cation:
                raise ValueError(f"Transition source is not cation occupied: {transition.label}")
            if transition.target not in virtual_cation:
                raise ValueError(f"Transition target is not cation virtual: {transition.label}")
            matrix[virtual_cation.index(transition.target), occ_cation.index(transition.source)] += transition.amplitude
            kept += 1
        coefficients[state.state] = matrix
        kept_counts[state.state] = kept
    return coefficients, kept_counts


def compute_dyson_orbitals(
    neutral_lines: list[str],
    cation_lines: list[str],
    neutral: OrbitalData,
    cation: OrbitalData,
    states: list[TDAState],
    program: str,
    amplitude_cutoff: float,
) -> list[DysonState]:
    if neutral.n_alpha + neutral.n_beta != cation.n_alpha + cation.n_beta + 1:
        raise ValueError("The neutral must have exactly one more electron than the cation")

    neutral_energy = parse_scf_energy(neutral_lines, program)
    cation_energy = parse_scf_energy(cation_lines, program)
    ground_ip_ev = (cation_energy - neutral_energy) * HARTREE_TO_EV
    excitation_by_state = {state.state: state.excitation_ev for state in states}

    coefficients, _kept_counts = build_cis_coefficients(states, cation, amplitude_cutoff)
    overlap_mo = spin_orbital_overlap(neutral, cation)
    nmo = neutral.nbf
    occ_neutral = list(range(neutral.n_alpha)) + list(range(nmo, nmo + neutral.n_beta))
    occ_cation = list(range(cation.n_alpha)) + list(range(nmo, nmo + cation.n_beta))
    dyson_mo = dyson_ks(coefficients, overlap_mo, {0: occ_neutral, +1: occ_cation})

    results: list[DysonState] = []
    for state in sorted(dyson_mo):
        coeff = dyson_mo[state]
        alpha_coeff = coeff[:nmo]
        beta_coeff = coeff[nmo:]
        alpha_ao = neutral.alpha_mo @ alpha_coeff
        beta_ao = neutral.beta_mo @ beta_coeff
        alpha_norm = float(np.sqrt(max(alpha_ao @ neutral.overlap @ alpha_ao, 0.0)))
        beta_norm = float(np.sqrt(max(beta_ao @ neutral.overlap @ beta_ao, 0.0)))
        total_norm = float(np.sqrt(alpha_norm**2 + beta_norm**2))
        excitation_ev = 0.0 if state == 0 else excitation_by_state[state]
        dominant_spin = "alpha" if alpha_norm >= beta_norm else "beta"
        results.append(
            DysonState(
                state=state,
                excitation_ev=excitation_ev,
                vertical_ip_ev=ground_ip_ev + excitation_ev,
                mo_coefficients=coeff,
                alpha_ao=alpha_ao,
                beta_ao=beta_ao,
                total_norm=total_norm,
                alpha_norm=alpha_norm,
                beta_norm=beta_norm,
                dominant_spin=dominant_spin,
            )
        )
    return results


def format_fchk_values(values: np.ndarray, values_per_line: int = 5) -> list[str]:
    lines: list[str] = []
    flat = np.asarray(values, dtype=float).ravel()
    for start in range(0, len(flat), values_per_line):
        chunk = flat[start : start + values_per_line]
        lines.append("".join(f"{value:16.8E}" for value in chunk) + "\n")
    return lines


def format_fchk_int_values(values: np.ndarray, values_per_line: int = 6) -> list[str]:
    lines: list[str] = []
    flat = np.asarray(values, dtype=int).ravel()
    for start in range(0, len(flat), values_per_line):
        chunk = flat[start : start + values_per_line]
        lines.append("".join(f"{int(value):12d}" for value in chunk) + "\n")
    return lines


def format_fchk_scalar(label: str, value: float) -> str:
    return f"{label[:43]:<43}R   {value:22.15E}\n"


def format_fchk_int_scalar(label: str, value: int) -> str:
    return f"{label[:43]:<43}I{int(value):17d}\n"


def format_fchk_array(label: str, values: np.ndarray) -> list[str]:
    return [f"{label[:43]:<43}R   N={len(values):12d}\n", *format_fchk_values(values)]


def format_fchk_int_array(label: str, values: np.ndarray) -> list[str]:
    return [f"{label[:43]:<43}I   N={len(values):12d}\n", *format_fchk_int_values(values)]


def _array_value_span(lines: list[str], label: str) -> tuple[int, int, int]:
    for idx, line in enumerate(lines):
        header = _field_header(line)
        if header is None:
            continue
        found_label, _type_code, rest = header
        if found_label != label:
            continue
        match = re.search(r"N=\s*(\d+)", rest)
        if match is None:
            raise ValueError(f"Field {label!r} is not an array")
        n_values = int(match.group(1))
        cursor = idx + 1
        seen = 0
        while seen < n_values and cursor < len(lines):
            seen += len(lines[cursor].split())
            cursor += 1
        if seen != n_values:
            raise ValueError(f"Field {label!r} has malformed array data")
        return idx + 1, cursor, n_values
    raise KeyError(f"Field not found in fchk: {label}")


def replace_fchk_array(lines: list[str], label: str, values: np.ndarray) -> None:
    start, end, n_values = _array_value_span(lines, label)
    flat = np.asarray(values, dtype=float).ravel()
    if len(flat) != n_values:
        raise ValueError(f"Replacement for {label!r} has {len(flat)} values, expected {n_values}")
    lines[start:end] = format_fchk_values(flat)


def insert_before_label(lines: list[str], label: str, new_lines: list[str]) -> None:
    for idx, line in enumerate(lines):
        header = _field_header(line)
        if header is not None and header[0] == label:
            lines[idx:idx] = new_lines
            return
    raise KeyError(f"Could not find insertion point {label!r}")


def delete_fchk_sections(lines: list[str], labels: set[str]) -> list[str]:
    cleaned: list[str] = []
    idx = 0
    while idx < len(lines):
        header = _field_header(lines[idx])
        if header is None or header[0] not in labels:
            cleaned.append(lines[idx])
            idx += 1
            continue

        idx += 1
        while idx < len(lines):
            next_header = _field_header(lines[idx])
            if next_header is not None:
                break
            idx += 1
    return cleaned


def dyson_columns_for_fchk(
    base_mo: np.ndarray,
    dyson_states: list[DysonState],
    spin: str,
    normalize: bool,
    dominant_component: bool = False,
) -> np.ndarray:
    matrix = np.array(base_mo, copy=True)
    for col, state in enumerate(dyson_states[: matrix.shape[1]]):
        if dominant_component:
            values = state.alpha_ao if state.dominant_spin == "alpha" else state.beta_ao
            norm = state.alpha_norm if state.dominant_spin == "alpha" else state.beta_norm
        else:
            values = state.alpha_ao if spin == "alpha" else state.beta_ao
            norm = state.alpha_norm if spin == "alpha" else state.beta_norm
        if normalize and norm > 1.0e-12:
            values = values / norm
        matrix[:, col] = values
    return matrix


def write_dyson_fchk(
    template: OrbitalData,
    output_fchk: str | Path,
    dyson_states: list[DysonState],
    normalize_for_view: bool = False,
    eom_ip_compat: bool = True,
) -> None:
    if template.lines is None:
        raise ValueError("No fchk template lines are available for output")
    lines = delete_fchk_sections(
        list(template.lines),
        {
            "Info1-9",
            "Full Title",
            "Route",
            "Number of symbols in /Mol/",
            "Atom Types",
            "Force Field",
            "ONIOM Charges",
            "ONIOM Multiplicities",
            "Atom Layers",
            "Atom Modifiers",
            "Atom Modified Types",
            "Link Atoms",
        },
    )
    if lines:
        lines[0] = "TDDFT Dyson orbitals\n"

    dyson_records: list[str] = []
    for eom_index, state in enumerate(dyson_states, start=1):
        component = state.beta_ao if state.dominant_spin == "beta" else state.alpha_ao
        component_norm = state.beta_norm if state.dominant_spin == "beta" else state.alpha_norm
        write_component = component
        if normalize_for_view and component_norm > 1.0e-12:
            write_component = component / component_norm
        if eom_ip_compat:
            ref_label = f"Ref - EOM-IP {eom_index}/A (alpha)"
        else:
            ref_label = f"Ref - TDDFT {state.state} ({state.dominant_spin})"
        dyson_records.append(format_fchk_scalar(ref_label, state.vertical_ip_ev / HARTREE_TO_EV))
        dyson_records.extend(format_fchk_array("Dyson Orbital (left)", write_component))
        dyson_records.extend(format_fchk_array("Dyson Orbital (right)", write_component))

    insert_before_label(lines, "Alpha MO coefficients", dyson_records)

    alpha_columns = dyson_columns_for_fchk(
        template.alpha_mo,
        dyson_states,
        "alpha",
        normalize_for_view,
        dominant_component=eom_ip_compat,
    )
    beta_columns = dyson_columns_for_fchk(template.beta_mo, dyson_states, "beta", normalize_for_view)
    replace_fchk_array(lines, "Alpha MO coefficients", alpha_columns.reshape(-1, order="F"))
    replace_fchk_array(lines, "Beta MO coefficients", beta_columns.reshape(-1, order="F"))

    alpha_energies = np.array(template.alpha_energies, copy=True)
    beta_energies = np.array(template.beta_energies, copy=True)
    for idx, state in enumerate(dyson_states[: template.nbf]):
        alpha_energies[idx] = state.vertical_ip_ev / HARTREE_TO_EV
        beta_energies[idx] = state.vertical_ip_ev / HARTREE_TO_EV
    replace_fchk_array(lines, "Alpha Orbital Energies", alpha_energies)
    replace_fchk_array(lines, "Beta Orbital Energies", beta_energies)
    Path(output_fchk).write_text("".join(lines))


def _molden_shell_label(shell_type: int) -> str:
    return {
        -1: "sp",
        0: "s",
        1: "p",
        -2: "d",
        2: "d",
        -3: "f",
        3: "f",
        -4: "g",
        4: "g",
    }.get(shell_type, "")


def _molden_primitive_normalization(exponent: float, angular_momentum: int) -> float:
    return (2.0 * exponent / pi) ** 0.75 * (4.0 * exponent) ** (0.5 * angular_momentum)


def _molden_contraction_coefficient(
    coefficient: float,
    exponent: float,
    angular_momentum: int,
    already_molden_normalized: bool,
) -> float:
    if already_molden_normalized:
        return coefficient
    return coefficient * _molden_primitive_normalization(exponent, angular_momentum)


def _shell_angular_momentum(shell_type: int) -> int:
    if shell_type == -1:
        raise ValueError("SP shells have separate s and p angular momenta")
    return abs(shell_type)


def write_dyson_molden(
    template: OrbitalData,
    output_molden: str | Path,
    dyson_states: list[DysonState],
    normalize_for_view: bool = False,
) -> None:
    if template.atomic_numbers is None or template.coordinates_bohr is None or not template.shells:
        raise ValueError("Cannot write Molden output without atoms and basis shells")

    coordinates = np.asarray(template.coordinates_bohr, dtype=float).reshape((-1, 3))
    lines: list[str] = [
        "[Molden Format]\n",
        "[Title]\n",
        " TDDFT/TDA Dyson orbitals\n",
        "\n",
        "[Atoms] AU\n",
    ]

    for idx, (atomic_number, xyz) in enumerate(zip(template.atomic_numbers, coordinates), start=1):
        symbol = Z_ELEMENT.get(int(atomic_number), f"X{int(atomic_number)}")
        lines.append(
            f"{symbol:<3s}{idx:6d}{int(atomic_number):6d}"
            f"{xyz[0]:20.10f}{xyz[1]:20.10f}{xyz[2]:20.10f}\n"
        )

    lines.append("[GTO]\n")
    shells_by_atom: dict[int, list[Shell]] = {}
    for shell in template.shells:
        shells_by_atom.setdefault(shell.atom_index, []).append(shell)

    for atom_idx in range(1, len(template.atomic_numbers) + 1):
        lines.append(f"{atom_idx:4d} 0\n")
        for shell in shells_by_atom.get(atom_idx, []):
            shell_label = _molden_shell_label(shell.shell_type)
            if not shell_label:
                raise ValueError(f"Unsupported shell type for Molden output: {shell.shell_type}")
            lines.append(f"{shell_label:<2s}{len(shell.exponents):4d} 1.0\n")
            for primitive_idx, exponent in enumerate(shell.exponents):
                coefficient = shell.coefficients[primitive_idx]
                if shell.shell_type == -1:
                    if shell.p_coefficients is None:
                        raise ValueError("SP shell is missing p contraction coefficients")
                    s_coefficient = _molden_contraction_coefficient(
                        coefficient,
                        exponent,
                        0,
                        template.molden_normalized_basis,
                    )
                    p_coefficient = _molden_contraction_coefficient(
                        shell.p_coefficients[primitive_idx],
                        exponent,
                        1,
                        template.molden_normalized_basis,
                    )
                    lines.append(
                        f"{exponent:20.10f}{s_coefficient:20.10f}"
                        f"{p_coefficient:20.10f}\n"
                    )
                else:
                    coefficient = _molden_contraction_coefficient(
                        coefficient,
                        exponent,
                        _shell_angular_momentum(shell.shell_type),
                        template.molden_normalized_basis,
                    )
                    lines.append(f"{exponent:20.10f}{coefficient:20.10f}\n")
        lines.append("\n")

    shell_types = {shell.shell_type for shell in template.shells}
    if -2 in shell_types:
        lines.append("[5D]\n")
    elif 2 in shell_types:
        lines.append("[6D]\n")
    if -3 in shell_types:
        lines.append("[7F]\n")
    elif 3 in shell_types:
        lines.append("[10F]\n")
    if -4 in shell_types:
        lines.append("[9G]\n")
    elif 4 in shell_types:
        lines.append("[15G]\n")

    lines.append("[MO]\n")
    for idx, state in enumerate(dyson_states, start=1):
        component = state.beta_ao if state.dominant_spin == "beta" else state.alpha_ao
        component_norm = state.beta_norm if state.dominant_spin == "beta" else state.alpha_norm
        if normalize_for_view and component_norm > 1.0e-12:
            component = component / component_norm
        spin_label = "Beta" if state.dominant_spin == "beta" else "Alpha"
        lines.append(f" Sym= Dyson_{state.state}\n")
        lines.append(f" Ene= {state.vertical_ip_ev / HARTREE_TO_EV: .12E}\n")
        lines.append(f" Spin= {spin_label}\n")
        lines.append(f" Occup= {state.total_norm ** 2:.6f}\n")
        for basis_idx, coefficient in enumerate(component, start=1):
            lines.append(f"{basis_idx:5d}{coefficient:20.12f}\n")

    Path(output_molden).write_text("".join(lines))


def write_report(
    output_report: str | Path,
    dyson_states: list[DysonState],
    program: str,
    amplitude_cutoff: float,
    normalize_for_view: bool,
    eom_ip_compat: bool,
) -> None:
    lines = [
        f"TDDFT/TDA Dyson orbital report ({program})\n",
        f"Amplitude cutoff: {amplitude_cutoff:.6g}\n",
        f"Written orbital coefficients normalized for viewing: {normalize_for_view}\n",
        f"IQmol EOM-IP compatibility labels: {eom_ip_compat}\n",
        "\n",
        "MO column  FCHK label  State  Excitation(eV)  Vertical IP(eV)  Norm  AlphaNorm  BetaNorm  DominantSpin\n",
    ]
    for col, state in enumerate(dyson_states, start=1):
        lines.append(
            f"{col:9d}  EOM-IP {col:2d}/A  {state.state:5d}  {state.excitation_ev:14.6f}  "
            f"{state.vertical_ip_ev:15.6f}  {state.total_norm:8.5f}  "
            f"{state.alpha_norm:9.5f}  {state.beta_norm:8.5f}  {state.dominant_spin}\n"
        )
    Path(output_report).write_text("".join(lines))


def print_summary(
    dyson_states: list[DysonState],
    output_fchk: Path | None,
    output_report: Path | None,
    output_molden: Path | None,
) -> None:
    if output_fchk is not None:
        print(f"Wrote {output_fchk}")
    if output_molden is not None:
        print(f"Wrote {output_molden}")
    if output_report is not None:
        print(f"Wrote {output_report}")
    print()
    print("State  Excitation(eV)  Vertical IP(eV)  Norm    Alpha    Beta   Spin")
    for state in dyson_states:
        print(
            f"{state.state:5d}  {state.excitation_ev:14.4f}  "
            f"{state.vertical_ip_ev:15.4f}  {state.total_norm:7.4f}  "
            f"{state.alpha_norm:7.4f}  {state.beta_norm:7.4f}  {state.dominant_spin}"
        )


def _required_path(value: str | Path | None, option_name: str) -> Path:
    if value is None:
        raise ValueError(f"Missing required input: {option_name}")
    return Path(value)


def load_inputs(
    program: str,
    *,
    neutral_out: str | Path | None,
    cation_out: str | Path | None,
    neutral_fchk: str | Path | None = None,
    cation_fchk: str | Path | None = None,
    neutral_checkpoint: str | Path | None = None,
    cation_checkpoint: str | Path | None = None,
    neutral_gbw: str | Path | None = None,
    cation_gbw: str | Path | None = None,
    neutral_molden: str | Path | None = None,
    cation_molden: str | Path | None = None,
    formchk_command: str = "formchk",
    orca_2mkl_command: str = "orca_2mkl",
) -> tuple[list[str], list[str], OrbitalData, OrbitalData, list[TDAState]]:
    program = program.lower()
    neutral_out = _required_path(neutral_out, "--neutral-out/-nout")
    cation_out = _required_path(cation_out, "--cation-out/-iout")
    neutral_lines = neutral_out.read_text().splitlines()
    cation_lines = cation_out.read_text().splitlines()

    if program == "qchem":
        neutral = read_fchk(_required_path(neutral_fchk, "--neutral-fchk/-nfchk"))
        cation = read_fchk(_required_path(cation_fchk, "--cation-fchk/-ifchk"))
        states = parse_qchem_tda_states(cation_lines, cation)
    elif program == "gaussian":
        neutral_fchk = ensure_gaussian_fchk(
            neutral_checkpoint,
            neutral_fchk,
            formchk_command,
        )
        cation_fchk = ensure_gaussian_fchk(
            cation_checkpoint,
            cation_fchk,
            formchk_command,
        )
        neutral = read_fchk(neutral_fchk)
        cation = read_fchk(cation_fchk)
        states = parse_gaussian_tda_states(cation_lines, cation)
    elif program == "orca":
        neutral_molden = ensure_orca_molden(
            neutral_gbw,
            neutral_molden,
            orca_2mkl_command,
        )
        cation_molden = ensure_orca_molden(
            cation_gbw,
            cation_molden,
            orca_2mkl_command,
        )
        neutral_charge, neutral_mult = parse_orca_charge_multiplicity(neutral_lines)
        cation_charge, cation_mult = parse_orca_charge_multiplicity(cation_lines)
        neutral = parse_molden(neutral_molden, neutral_charge, neutral_mult)
        cation = parse_molden(cation_molden, cation_charge, cation_mult)
        shared_overlap = infer_overlap_from_mos(neutral.alpha_mo)
        neutral.overlap = shared_overlap
        cation.overlap = shared_overlap
        attach_minimal_fchk_lines(neutral, "ORCA TDDFT Dyson orbitals")
        states = parse_orca_tda_states(cation_lines, cation)
    else:
        raise ValueError(f"Unsupported program: {program}")

    return neutral_lines, cation_lines, neutral, cation, states


def run_calculation(
    *,
    program: str,
    neutral_out: str | Path | None,
    cation_out: str | Path | None,
    neutral_fchk: str | Path | None = None,
    cation_fchk: str | Path | None = None,
    neutral_checkpoint: str | Path | None = None,
    cation_checkpoint: str | Path | None = None,
    neutral_gbw: str | Path | None = None,
    cation_gbw: str | Path | None = None,
    neutral_molden: str | Path | None = None,
    cation_molden: str | Path | None = None,
    output_fchk: str | Path | None = "tddft_dyson.fchk",
    output_report: str | Path | None = "tddft_dyson_report.txt",
    output_molden: str | Path | None = "tddft_dyson.molden",
    write_fchk_file: bool = True,
    write_molden_file: bool = True,
    formchk_command: str = "formchk",
    orca_2mkl_command: str = "orca_2mkl",
    amplitude_cutoff: float = 0.01,
    normalize_for_view: bool = False,
    eom_ip_compat: bool = True,
) -> CalculationResult:
    program = program.lower()
    neutral_lines, cation_lines, neutral, cation, states = load_inputs(
        program,
        neutral_out=neutral_out,
        cation_out=cation_out,
        neutral_fchk=neutral_fchk,
        cation_fchk=cation_fchk,
        neutral_checkpoint=neutral_checkpoint,
        cation_checkpoint=cation_checkpoint,
        neutral_gbw=neutral_gbw,
        cation_gbw=cation_gbw,
        neutral_molden=neutral_molden,
        cation_molden=cation_molden,
        formchk_command=formchk_command,
        orca_2mkl_command=orca_2mkl_command,
    )

    dyson_states = compute_dyson_orbitals(
        neutral_lines=neutral_lines,
        cation_lines=cation_lines,
        neutral=neutral,
        cation=cation,
        states=states,
        program=program,
        amplitude_cutoff=amplitude_cutoff,
    )

    output_fchk_path = Path(output_fchk) if write_fchk_file and output_fchk is not None else None
    output_report_path = Path(output_report) if output_report is not None else None
    output_molden_path = (
        Path(output_molden) if write_molden_file and output_molden is not None else None
    )

    if output_fchk_path is not None:
        output_fchk_path.parent.mkdir(parents=True, exist_ok=True)
        write_dyson_fchk(
            template=neutral,
            output_fchk=output_fchk_path,
            dyson_states=dyson_states,
            normalize_for_view=normalize_for_view,
            eom_ip_compat=eom_ip_compat,
        )

    if output_molden_path is not None:
        output_molden_path.parent.mkdir(parents=True, exist_ok=True)
        write_dyson_molden(
            template=neutral,
            output_molden=output_molden_path,
            dyson_states=dyson_states,
            normalize_for_view=normalize_for_view,
        )

    if output_report_path is not None:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        write_report(
            output_report=output_report_path,
            dyson_states=dyson_states,
            program=program,
            amplitude_cutoff=amplitude_cutoff,
            normalize_for_view=normalize_for_view,
            eom_ip_compat=eom_ip_compat,
        )

    return CalculationResult(
        dyson_states=dyson_states,
        output_fchk=output_fchk_path,
        output_report=output_report_path,
        output_molden=output_molden_path,
    )
