"""CLI helpers — argument parsing and table formatting."""

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run parallel benchmarks with selected methods')
    parser.add_argument('--methods', '-m',
                       nargs='+',
                       default=['standard', 'cr_gnn', 'nrgnn'],
                       help='Methods to benchmark (space separated)')
    parser.add_argument('--run-id', '-r',
                       type=int,
                       default=1,
                       help='Fixed run ID for consistent seeds (default: 1)')
    return parser.parse_args()


def print_table(headers, rows, col_widths):

    def separator():
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def row_line(row):
        return "|" + "|".join(f" {str(val).ljust(w)} " for val, w in zip(row, col_widths)) + "|"

    print(separator())
    print(row_line(headers))
    print(separator())
    for row in rows:
        print(row_line(row))
        print(separator())
