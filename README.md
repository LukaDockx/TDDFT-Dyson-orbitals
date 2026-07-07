# TDDFT Dyson Orbitals

`tddft-dyson` computes TDDFT/TDA Dyson orbitals from separately run neutral and cation calculations. It currently supports Q-Chem, Gaussian 16, and ORCA text/checkpoint-style outputs.

The code writes:

- an IQmol-compatible `.fchk` file
- a Molden file containing the Dyson orbitals and basis set
- a text report with excitation energies, vertical IPs, norms, and dominant spin components

The determinant-overlap expression follows the supporting information in `Supporting_info_Dyson.pdf`.

## Install

From this directory:

```bash
python -m pip install -e .
```

That installs the command:

```bash
TDDFT_DO --help
```

Without installation, run from the checkout with:

```bash
PYTHONPATH=src python -m tddft_dyson --help
```

The old script name is kept as a wrapper:

```bash
python Dyson_tddft.py --help
```

## Required Inputs

All programs need the neutral and cation output files because the script reads SCF energies from them. The cation output must also contain the TDDFT/TDA excitation amplitudes.

Q-Chem:

- neutral output: `-nout` / `--neutral-out`
- cation output with canonical TDA amplitudes: `-iout` / `--cation-out`
- neutral fchk: `-nfchk` / `--neutral-fchk`
- cation fchk: `-ifchk` / `--cation-fchk`

Gaussian 16:

- neutral log: `-nout`
- cation log with the printed excitation vector: `-iout`
- neutral/cation `.fchk` via `-nfchk` and `-ifchk`
- alternatively, pass `.chk` files with `-nchk` and `-ichk`; the script will call `formchk`

ORCA:

- neutral output: `-nout`
- cation output with TDA excited states: `-iout`
- neutral/cation Molden files via `-nmolden` and `-imolden`
- alternatively, pass `.gbw` files with `-ngbw` and `-igbw`; the script will call `orca_2mkl -molden`

## Basic CLI Usage

Both equals signs and whitespace work:

```bash
TDDFT_DO -p=qchem -nout=neutral.out -iout=cation.out -nfchk=neutral.fchk -ifchk=cation.fchk
```

```bash
TDDFT_DO -p qchem -nout neutral.out -iout cation.out -nfchk neutral.fchk -ifchk cation.fchk
```

Default outputs are:

- `tddft_dyson.fchk`
- `tddft_dyson.molden`
- `tddft_dyson_report.txt`

Useful output options:

```bash
TDDFT_DO ... -o dyson_iqmol.fchk -m dyson.molden -r dyson_report.txt
TDDFT_DO ... --normalize-for-view
TDDFT_DO ... --no-fchk
TDDFT_DO ... --no-molden
```

`--normalize-for-view` normalizes each written orbital for visualization. The report still prints the raw Dyson norms.

## Examples

The `examples/` directory contains portable methanol examples. Large binary scratch/checkpoint files are intentionally not required; Gaussian examples use `.fchk`, and ORCA examples use Molden files.

Q-Chem:

```bash
TDDFT_DO \
  -p qchem \
  -nout examples/qchem_methanol/MeOH_n.out \
  -iout examples/qchem_methanol/MeOH_i.out \
  -nfchk examples/qchem_methanol/MeOH_n.in.fchk \
  -ifchk examples/qchem_methanol/MeOH_i.in.fchk \
  -o examples/qchem_methanol/MeOH_tddft_dyson.fchk \
  -m examples/qchem_methanol/MeOH_tddft_dyson.molden \
  -r examples/qchem_methanol/MeOH_tddft_dyson_report.txt
```

Gaussian 16:

```bash
TDDFT_DO \
  -p gaussian \
  -nout examples/gaussian_methanol/MeOH_n.log \
  -iout examples/gaussian_methanol/MeOH_i.log \
  -nfchk examples/gaussian_methanol/MeOH_n.fchk \
  -ifchk examples/gaussian_methanol/MeOH_i.fchk \
  -o examples/gaussian_methanol/MeOH_tddft_dyson.fchk \
  -m examples/gaussian_methanol/MeOH_tddft_dyson.molden \
  -r examples/gaussian_methanol/MeOH_tddft_dyson_report.txt
```

ORCA:

```bash
TDDFT_DO \
  -p orca \
  -nout examples/orca_methanol/MeOH_n.out \
  -iout examples/orca_methanol/MeOH_i.out \
  -nmolden examples/orca_methanol/MeOH_n.molden.input \
  -imolden examples/orca_methanol/MeOH_i.molden.input \
  -o examples/orca_methanol/MeOH_tddft_dyson.fchk \
  -m examples/orca_methanol/MeOH_tddft_dyson.molden \
  -r examples/orca_methanol/MeOH_tddft_dyson_report.txt
```

## Python Interface

```python
from tddft_dyson import run_calculation

result = run_calculation(
    program="qchem",
    neutral_out="examples/qchem_methanol/MeOH_n.out",
    cation_out="examples/qchem_methanol/MeOH_i.out",
    neutral_fchk="examples/qchem_methanol/MeOH_n.in.fchk",
    cation_fchk="examples/qchem_methanol/MeOH_i.in.fchk",
    output_fchk="dyson.fchk",
    output_molden="dyson.molden",
    output_report="dyson_report.txt",
)

for state in result.dyson_states:
    print(state.state, state.vertical_ip_ev, state.total_norm)
```

## IQmol Notes

IQmol recognizes Dyson orbital surface records from Q-Chem EOM-IP calculations. For that reason, the `.fchk` writer labels the TDDFT Dyson records as `Ref - EOM-IP ...` and writes `Dyson Orbital (left/right)` sections. These are TDDFT/TDA Dyson orbitals; the EOM-IP naming is only an IQmol compatibility convention.

The same dominant Dyson component is also written into the first alpha MO columns so IQmol can plot them from the canonical orbital surface menu if needed. The report maps MO column numbers back to TDDFT state numbers.

For easier visual inspection, use:

```bash
TDDFT_DO ... --normalize-for-view
```

The Molden output is written by default and can be opened in programs that support Molden orbitals.
