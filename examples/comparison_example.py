from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from multilevel_sabre import MultiLevelSabre
import time
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Iterable

from util import EAGLE_COUPLING, sabre, count_swaps

def run_comparison_example(qasm_path: Path) -> Dict:
    """ Run a comparison between SABRE and MultiLevel SABRE on a given QASM circuit.
    
    Args:
        qasm_path (Path): Path to the QASM file containing the quantum circuit.
        
    Returns:
        Dict: A dictionary containing the results of the comparison.
    """

    # Load the QASM circuit
    circuit = QuantumCircuit.from_qasm_file(str(qasm_path))
    
    # Create the EAGLE coupling map
    coupling_map = CouplingMap(couplinglist=EAGLE_COUPLING)
    coupling_map.make_symmetric()

    # Run SABRE
    print("\nRunning SABRE...")
    start_time = time.time()
    sabre_swaps, sabre_circuit = sabre(
        circuit=circuit,
        coupling=EAGLE_COUPLING,
        random_seed=1
    )
    sabre_time = time.time() - start_time
    sabre_ops = sabre_circuit.count_ops()

    # Run MultiLevel SABRE
    print("\nRunning MultiLevel SABRE...")
    start_time = time.time()
    multilevel_pass = PassManager([
        MultiLevelSabre(
            coupling_graph=coupling_map,
            cycles=10,
            random_seed=1,
            coarsest_solving_trials=50,
            num_interpolation=10,
            use_initial_embedding=True,
            verbose=0
        )
    ])
    multilevel_circuit = multilevel_pass.run(circuit)
    multilevel_time = time.time() - start_time
    multilevel_ops = multilevel_circuit.count_ops()

    # Count SWAPs in MultiLevel SABRE result
    multilevel_swaps = count_swaps(multilevel_circuit)

    # Prepare metrics row (keep ops as JSON strings for CSV-friendliness)
    row = {
        "qasm_file": str(qasm_path),
        "sabre_swaps": sabre_swaps,
        "sabre_time_s": round(sabre_time, 4),
        "sabre_depth": sabre_circuit.depth(),
        "sabre_2q_depth": sabre_circuit.depth(lambda x: x.operation.num_qubits == 2),
        "sabre_size": sabre_circuit.size(),
        "sabre_ops_json": json.dumps(sabre_ops, default=int),

        "multilevel_swaps": multilevel_swaps,
        "multilevel_time_s": round(multilevel_time, 4),
        "multilevel_depth": multilevel_circuit.depth(),
        "multilevel_2q_depth": multilevel_circuit.depth(lambda x: x.operation.num_qubits == 2),
        "multilevel_size": multilevel_circuit.size(),
        "multilevel_ops_json": json.dumps(multilevel_ops, default=int),
    }

    # Derived metrics
    row["speedup"] = round(row["sabre_time_s"] / row["multilevel_time_s"], 4) if row["multilevel_time_s"] > 0 else float("inf")
    row["swap_reduction_pct"] = round(((row["sabre_swaps"] - row["multilevel_swaps"]) / row["sabre_swaps"] * 100), 3) if row["sabre_swaps"] else 0.0

    # Pretty print to terminal
    print("\nComparison Results:")
    print("SABRE:")
    print(f"  - Number of SWAPs: {row['sabre_swaps']}")
    print(f"  - Compilation time: {row['sabre_time_s']:.2f} seconds")
    print(f"  - Circuit depth: {row['sabre_depth']}")
    print(f"  - Circuit 2Q depth: {row['sabre_2q_depth']}")
    print(f"  - Circuit size: {row['sabre_size']}")
    print("\nMultiLevel SABRE:")
    print(f"  - Number of SWAPs: {row['multilevel_swaps']}")
    print(f"  - Compilation time: {row['multilevel_time_s']:.2f} seconds")
    print(f"  - Circuit depth: {row['multilevel_depth']}")
    print(f"  - Circuit 2Q depth: {row['multilevel_2q_depth']}")
    print(f"  - Circuit size: {row['multilevel_size']}")
    print(f"\nSpeedup: {row['speedup']:.2f}x")
    print(f"SWAP reduction: {row['swap_reduction_pct']:.1f}%")

    return row

def gather_qasm_files(paths: Iterable[str]) -> List[Path]:
    """Expand a list of files/dirs into a sorted list of .qasm files."""
    result: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            result.extend(sorted(path.glob("*.qasm")))
        elif path.is_file() and path.suffix.lower() == ".qasm":
            result.append(path)
        else:
            print(f"Skipping {p}: not a .qasm file or directory.")
    return sorted(set(result))

def write_csv(rows: List[Dict], out_csv: Path) -> None:
    if not rows:
        print("No rows to write.")
        return
    fieldnames = [
        "qasm_file",
        "sabre_swaps", "sabre_time_s", "sabre_depth", "sabre_2q_depth", "sabre_size", "sabre_ops_json",
        "multilevel_swaps", "multilevel_time_s", "multilevel_depth", "multilevel_2q_depth", "multilevel_size", "multilevel_ops_json",
        "speedup", "swap_reduction_pct",
        "error",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nWrote CSV: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SABRE vs MultiLevel SABRE on one or more QASM files or directories."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more QASM files and/or directories (e.g., examples/circuits or examples/circuits/adder_n118.qasm)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write a CSV of results (e.g., results/comparison.csv)",
    )
    args = parser.parse_args()

    qasm_files = gather_qasm_files(args.paths)
    if not qasm_files:
        raise SystemExit("No .qasm files found.")

    rows = []
    for q in qasm_files:
        try:
            rows.append(run_comparison_example(q))
        except Exception as e:
            # Keep going; record the error in the CSV
            print(f"[{q.name}] ERROR: {e}")
            rows.append({"qasm_file": str(q), "error": repr(e)})

    if args.csv:
        write_csv(rows, Path(args.csv))