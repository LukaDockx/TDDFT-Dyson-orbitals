from __future__ import annotations

import argparse
import shutil

from .core import print_summary, run_calculation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TDDFT_DO",
        description="Compute TDDFT/TDA Dyson orbitals from Q-Chem, Gaussian 16, or ORCA files.",
    )
    parser.add_argument(
        "-p",
        "--program",
        choices=["qchem", "gaussian", "orca"],
        required=True,
        help="Electronic-structure package that produced the input files.",
    )
    parser.add_argument("-nout", "--neutral-out", required=True, help="Neutral output/log file.")
    parser.add_argument(
        "-iout",
        "--cation-out",
        required=True,
        help="Cation TDDFT/TDA output/log file containing excitation amplitudes.",
    )
    parser.add_argument("-nfchk", "--neutral-fchk", help="Neutral formatted checkpoint file.")
    parser.add_argument("-ifchk", "--cation-fchk", help="Cation formatted checkpoint file.")
    parser.add_argument(
        "-nchk",
        "--neutral-checkpoint",
        help="Neutral Gaussian .chk file. Used with formchk if --neutral-fchk is absent.",
    )
    parser.add_argument(
        "-ichk",
        "--cation-checkpoint",
        help="Cation Gaussian .chk file. Used with formchk if --cation-fchk is absent.",
    )
    parser.add_argument(
        "-ngbw",
        "--neutral-gbw",
        help="Neutral ORCA .gbw file. Used with orca_2mkl if --neutral-molden is absent.",
    )
    parser.add_argument(
        "-igbw",
        "--cation-gbw",
        help="Cation ORCA .gbw file. Used with orca_2mkl if --cation-molden is absent.",
    )
    parser.add_argument("-nmolden", "--neutral-molden", help="Neutral Molden file.")
    parser.add_argument("-imolden", "--cation-molden", help="Cation Molden file.")
    parser.add_argument(
        "-o",
        "--output",
        "--output-fchk",
        dest="output_fchk",
        default="tddft_dyson.fchk",
        help="IQmol-compatible fchk output. Default: tddft_dyson.fchk.",
    )
    parser.add_argument(
        "-m",
        "--molden",
        dest="output_molden",
        default="tddft_dyson.molden",
        help="Molden output containing the Dyson orbitals. Default: tddft_dyson.molden.",
    )
    parser.add_argument(
        "-r",
        "--report",
        dest="output_report",
        default="tddft_dyson_report.txt",
        help="Text report output. Default: tddft_dyson_report.txt.",
    )
    parser.add_argument("--no-fchk", action="store_true", help="Do not write fchk output.")
    parser.add_argument("--no-molden", action="store_true", help="Do not write Molden output.")
    parser.add_argument("--no-report", action="store_true", help="Do not write a text report.")
    parser.add_argument(
        "--formchk",
        default=shutil.which("formchk") or "formchk",
        help="Gaussian formchk executable. Default: first formchk on PATH.",
    )
    parser.add_argument(
        "--orca-2mkl",
        default=shutil.which("orca_2mkl") or "orca_2mkl",
        help="ORCA orca_2mkl executable. Default: first orca_2mkl on PATH.",
    )
    parser.add_argument(
        "--amplitude-cutoff",
        type=float,
        default=0.01,
        help="Drop TDA amplitudes below this absolute value before contraction.",
    )
    parser.add_argument(
        "--normalize-for-view",
        action="store_true",
        help="Normalize written Dyson orbitals for visualization. Reported norms remain raw.",
    )
    parser.add_argument(
        "--tddft-labels",
        action="store_true",
        help="Write TDDFT fchk labels instead of IQmol-compatible EOM-IP labels.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_calculation(
        program=args.program,
        neutral_out=args.neutral_out,
        cation_out=args.cation_out,
        neutral_fchk=args.neutral_fchk,
        cation_fchk=args.cation_fchk,
        neutral_checkpoint=args.neutral_checkpoint,
        cation_checkpoint=args.cation_checkpoint,
        neutral_gbw=args.neutral_gbw,
        cation_gbw=args.cation_gbw,
        neutral_molden=args.neutral_molden,
        cation_molden=args.cation_molden,
        output_fchk=args.output_fchk,
        output_report=None if args.no_report else args.output_report,
        output_molden=args.output_molden,
        write_fchk_file=not args.no_fchk,
        write_molden_file=not args.no_molden,
        formchk_command=args.formchk,
        orca_2mkl_command=args.orca_2mkl,
        amplitude_cutoff=args.amplitude_cutoff,
        normalize_for_view=args.normalize_for_view,
        eom_ip_compat=not args.tddft_labels,
    )
    print_summary(
        result.dyson_states,
        result.output_fchk,
        result.output_report,
        result.output_molden,
    )


if __name__ == "__main__":
    main()
