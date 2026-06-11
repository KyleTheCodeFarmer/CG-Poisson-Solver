#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "benchmark.csv"
THREAD_CSV_PATH = ROOT / "results" / "thread_scaling.csv"
RUNTIME_PLOT = ROOT / "results" / "runtime_scaling.png"
ITERATION_PLOT = ROOT / "results" / "iteration_scaling.png"
THREAD_SPEEDUP_PLOT = ROOT / "results" / "thread_speedup.png"
THREAD_RUNTIME_PLOT = ROOT / "results" / "thread_runtime.png"
UNKNOWN_SOURCE_CSV = ROOT / "results" / "unknown_source.csv"
UNKNOWN_SOLUTION_CSV = ROOT / "results" / "unknown_solution.csv"
UNKNOWN_SOURCE_PLOT = ROOT / "results" / "unknown_source.png"
UNKNOWN_SOLUTION_PLOT = ROOT / "results" / "unknown_solution.png"


def read_benchmark_csv(path):
    data = {
        "N": [],
        "cells": [],
        "cg_iterations": [],
        "cg_time": [],
        "sor_iterations": [],
        "sor_time": [],
    }

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data["N"].append(int(row["N"]))
            data["cells"].append(int(row["cells"]))
            data["cg_iterations"].append(int(row["cg_iterations"]))
            data["cg_time"].append(float(row["cg_time"]))
            data["sor_iterations"].append(int(row["sor_iterations"]))
            data["sor_time"].append(float(row["sor_time"]))

    return data


def read_thread_csv(path):
    data = {
        "threads": [],
        "cg_time": [],
        "cg_speedup": [],
    }

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data["threads"].append(int(row["threads"]))
            data["cg_time"].append(float(row["cg_time"]))
            data["cg_speedup"].append(float(row["cg_speedup"]))

    return data


def read_field_csv(path):
    field = []

    with path.open(newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            field.append([float(value) for value in row])

    return field


def add_grid_labels(cells, values, grid_sizes):
    for x, y, n in zip(cells, values, grid_sizes):
        plt.annotate(f"N={n}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)


def plot_runtime(data):
    plt.figure(figsize=(7.2, 4.8), dpi=160)
    plt.loglog(data["cells"], data["cg_time"], "o-", linewidth=2, color="blue", label="CG")
    plt.loglog(data["cells"], data["sor_time"], "s-", linewidth=2, color="red", label="SOR")
    add_grid_labels(data["cells"], data["cg_time"], data["N"])

    plt.xlabel("Number of cells (N x N)")
    plt.ylabel("Wall-clock time (s)")
    plt.title("Runtime Scaling: CG vs SOR")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUNTIME_PLOT)


def plot_iterations(data):
    plt.figure(figsize=(7.2, 4.8), dpi=160)
    plt.loglog(data["cells"], data["cg_iterations"], "o-", linewidth=2, color="blue", label="CG")
    plt.loglog(data["cells"], data["sor_iterations"], "s-", linewidth=2, color="red", label="SOR")
    add_grid_labels(data["cells"], data["sor_iterations"], data["N"])

    plt.xlabel("Number of cells (N x N)")
    plt.ylabel("Iterations")
    plt.title("Iteration Scaling: CG vs SOR")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ITERATION_PLOT)


def plot_thread_speedup(data):
    plt.figure(figsize=(7.2, 4.8), dpi=160)
    plt.plot(data["threads"], data["cg_speedup"], "o-", linewidth=2, color="blue", label="Measured")
    plt.plot(data["threads"], data["threads"], "--", linewidth=1.5, color="gray", label="Ideal")

    plt.xlabel("Number of OpenMP threads")
    plt.ylabel("CG speedup")
    plt.title("CG Thread Speedup")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(data["threads"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(THREAD_SPEEDUP_PLOT)


def plot_thread_runtime(data):
    plt.figure(figsize=(7.2, 4.8), dpi=160)
    plt.plot(data["threads"], data["cg_time"], "o-", linewidth=2, color="blue")

    plt.xlabel("Number of OpenMP threads")
    plt.ylabel("CG wall-clock time (s)")
    plt.title("CG Runtime vs Threads")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xticks(data["threads"])
    plt.tight_layout()
    plt.savefig(THREAD_RUNTIME_PLOT)


def plot_field(field, output_path, title, color_label):
    plt.figure(figsize=(6.0, 5.2), dpi=160)
    image = plt.imshow(field, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    cbar = plt.colorbar(image)
    cbar.set_label(color_label)
    plt.tight_layout()
    plt.savefig(output_path)


def main():
    data = read_benchmark_csv(CSV_PATH)
    thread_data = read_thread_csv(THREAD_CSV_PATH)

    plot_runtime(data)
    plot_iterations(data)
    plot_thread_speedup(thread_data)
    plot_thread_runtime(thread_data)

    if UNKNOWN_SOURCE_CSV.exists() and UNKNOWN_SOLUTION_CSV.exists():
        unknown_source = read_field_csv(UNKNOWN_SOURCE_CSV)
        unknown_solution = read_field_csv(UNKNOWN_SOLUTION_CSV)
        plot_field(unknown_source, UNKNOWN_SOURCE_PLOT, "Unknown Multi-Mode Source", "rhs")
        plot_field(unknown_solution, UNKNOWN_SOLUTION_PLOT, "Solved Potential Field", "phi")

    print(f"Wrote {RUNTIME_PLOT}")
    print(f"Wrote {ITERATION_PLOT}")
    print(f"Wrote {THREAD_SPEEDUP_PLOT}")
    print(f"Wrote {THREAD_RUNTIME_PLOT}")
    if UNKNOWN_SOURCE_CSV.exists() and UNKNOWN_SOLUTION_CSV.exists():
        print(f"Wrote {UNKNOWN_SOURCE_PLOT}")
        print(f"Wrote {UNKNOWN_SOLUTION_PLOT}")


if __name__ == "__main__":
    main()
