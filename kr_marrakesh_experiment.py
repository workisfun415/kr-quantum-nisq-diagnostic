"""
K-R Scaling Exponent Experiment — ibm_marrakesh
Author: Ramakrishna Pasupuleti
Date: July 2026

Fixed for IBM Quantum Open Plan (no Session allowed)
Runs jobs directly using SamplerV2

Experiments:
  1. GHZ qubit scaling: n = 2, 5, 10, 15, 20, 25, 30
  2. Depth scaling: d = 1, 2, 4, 8, 16, 32, 64
"""

# ─────────────────────────────────────────────
# STEP 0: IMPORTS
# ─────────────────────────────────────────────
import json
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ─────────────────────────────────────────────
# STEP 1: CONFIGURATION
# ─────────────────────────────────────────────
IBM_TOKEN    = "hspcz_fYcybtuxbdHomy0niDhPpTsmjomGNf7S0_kDzG"   # <-- paste your token
BACKEND_NAME = "ibm_marrakesh"
SHOTS        = 4096
OPTIMIZATION_LEVEL = 0  # Critical: must be 0

GHZ_QUBIT_LIST = [2, 5, 10, 15, 20, 25, 30]
DEPTH_LIST     = [1, 2, 4, 8, 16, 32, 64]
N_QUBITS_DEPTH = 5

# ─────────────────────────────────────────────
# STEP 2: CONNECT
# ─────────────────────────────────────────────
print("Connecting to IBM Quantum...")
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=IBM_TOKEN
)
backend = service.backend(BACKEND_NAME)
print(f"Connected to: {backend.name} ({backend.num_qubits} qubits)")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─────────────────────────────────────────────
# STEP 3: CIRCUIT BUILDERS
# ─────────────────────────────────────────────
def build_ghz_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def build_clifford_mirror_circuit(n_qubits, depth):
    import random
    random.seed(42 + depth)  # reproducible
    qc = QuantumCircuit(n_qubits)
    gate_sequence = []
    for _ in range(depth):
        layer = []
        for q in range(n_qubits):
            gate = random.choice(['h', 's', 'x', 'y', 'z'])
            layer.append((gate, q))
        gate_sequence.append(layer)
    # Forward
    for layer in gate_sequence:
        for (gate, q) in layer:
            getattr(qc, gate)(q)
    # Inverse
    for layer in reversed(gate_sequence):
        for (gate, q) in layer:
            if gate == 's':
                qc.sdg(q)
            else:
                getattr(qc, gate)(q)
    qc.measure_all()
    return qc


# ─────────────────────────────────────────────
# STEP 4: FIDELITY HELPERS
# ─────────────────────────────────────────────
def ghz_fidelity(counts, n_qubits):
    total     = sum(counts.values())
    all_zeros = '0' * n_qubits
    all_ones  = '1' * n_qubits
    p0 = counts.get(all_zeros, 0) / total
    p1 = counts.get(all_ones,  0) / total
    return p0 + p1, 1.0 - (p0 + p1)


def mirror_fidelity(counts, n_qubits):
    total     = sum(counts.values())
    all_zeros = '0' * n_qubits
    f = counts.get(all_zeros, 0) / total
    return f, 1.0 - f


# ─────────────────────────────────────────────
# STEP 5: K-R FITTING
# ─────────────────────────────────────────────
def kr_model(K, C0, alpha, R):
    return C0 / (K ** alpha) + R


def fit_kr(x_values, infidelities):
    x = np.array(x_values, dtype=float)
    y = np.array(infidelities, dtype=float)
    K = 1.0 / x
    try:
        popt, _ = curve_fit(
            kr_model, K, y,
            p0=[0.1, 0.5, 0.01],
            bounds=([0, 0.01, 0], [50, 15, 2]),
            method='trf',
            maxfev=10000
        )
        C0, alpha, R = popt
        y_pred = kr_model(K, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"alpha": float(alpha), "C0": float(C0),
                "R": float(R), "R2": float(R2), "ok": True}
    except Exception as e:
        return {"alpha": None, "C0": None, "R": None,
                "R2": None, "ok": False, "error": str(e)}


# ─────────────────────────────────────────────
# STEP 6: HELPER — RUN ONE CIRCUIT
# ─────────────────────────────────────────────
def run_circuit(qc, label=""):
    """Transpile and run a single circuit. Returns counts and job_id."""
    print(f"  Transpiling {label}...")
    qc_t = transpile(qc, backend=backend,
                     optimization_level=OPTIMIZATION_LEVEL)
    sampler = Sampler(mode=backend)
    print(f"  Submitting {label}...")
    job = sampler.run([qc_t], shots=SHOTS)
    print(f"  Job ID: {job.job_id()} — waiting for result...")
    result     = job.result()
    pub_result = result[0]
    counts     = pub_result.data.meas.get_counts()
    print(f"  Done.")
    return counts, job.job_id()


# ─────────────────────────────────────────────
# STEP 7: EXPERIMENT 1 — GHZ QUBIT SCALING
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("EXPERIMENT 1: GHZ Qubit Scaling")
print(f"Qubits: {GHZ_QUBIT_LIST}")
print("="*50)

ghz_results = {}

for n in GHZ_QUBIT_LIST:
    print(f"\nRunning GHZ n={n}...")
    qc = build_ghz_circuit(n)
    counts, job_id = run_circuit(qc, label=f"GHZ n={n}")
    fidelity, infidelity = ghz_fidelity(counts, n)
    ghz_results[str(n)] = {
        "n_qubits":   n,
        "fidelity":   float(fidelity),
        "infidelity": float(infidelity),
        "job_id":     job_id
    }
    print(f"  n={n}: fidelity={fidelity:.4f}, infidelity={infidelity:.4f}")

ghz_n_list  = [int(k) for k in ghz_results]
ghz_inf     = [ghz_results[str(n)]["infidelity"] for n in ghz_n_list]
ghz_kr_fit  = fit_kr(ghz_n_list, ghz_inf)
print(f"\nGHZ K-R fit: alpha={ghz_kr_fit['alpha']:.4f}, "
      f"R2={ghz_kr_fit['R2']:.4f}")


# ─────────────────────────────────────────────
# STEP 8: EXPERIMENT 2 — DEPTH SCALING
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("EXPERIMENT 2: Clifford Mirror Depth Scaling")
print(f"Depths: {DEPTH_LIST}, Qubits: {N_QUBITS_DEPTH}")
print("="*50)

depth_results = {}

for d in DEPTH_LIST:
    print(f"\nRunning depth d={d}...")
    qc = build_clifford_mirror_circuit(N_QUBITS_DEPTH, d)
    counts, job_id = run_circuit(qc, label=f"depth d={d}")
    fidelity, infidelity = mirror_fidelity(counts, N_QUBITS_DEPTH)
    depth_results[str(d)] = {
        "depth":      d,
        "fidelity":   float(fidelity),
        "infidelity": float(infidelity),
        "job_id":     job_id
    }
    print(f"  d={d}: fidelity={fidelity:.4f}, infidelity={infidelity:.4f}")

depth_d_list  = [int(k) for k in depth_results]
depth_inf     = [depth_results[str(d)]["infidelity"] for d in depth_d_list]
depth_kr_fit  = fit_kr(depth_d_list, depth_inf)
print(f"\nDepth K-R fit: alpha={depth_kr_fit['alpha']:.4f}, "
      f"R2={depth_kr_fit['R2']:.4f}")


# ─────────────────────────────────────────────
# STEP 9: SAVE RESULTS
# ─────────────────────────────────────────────
output = {
    "backend":            BACKEND_NAME,
    "timestamp":          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "shots":              SHOTS,
    "optimization_level": OPTIMIZATION_LEVEL,

    "exp1_ghz_scaling": {
        "description": "GHZ qubit scaling n=2 to 30",
        "qubit_list":  GHZ_QUBIT_LIST,
        "data":        ghz_results,
        "kr_fit":      ghz_kr_fit
    },

    "exp2_depth_scaling": {
        "description": "Clifford mirror depth scaling d=1 to 64",
        "depth_list":  DEPTH_LIST,
        "n_qubits":    N_QUBITS_DEPTH,
        "data":        depth_results,
        "kr_fit":      depth_kr_fit
    },

    "fingerprint": {
        "depth_alpha": depth_kr_fit["alpha"],
        "depth_R2":    depth_kr_fit["R2"],
        "ghz_alpha":   ghz_kr_fit["alpha"],
        "ghz_R2":      ghz_kr_fit["R2"],
        "note":        "Third Heron r2 device fingerprint — ibm_marrakesh"
    }
}

output_file = "kr_marrakesh_results.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "="*50)
print("ALL EXPERIMENTS COMPLETE")
print("="*50)
print(f"Saved to: {output_file}")
print(f"\nSUMMARY — ibm_marrakesh fingerprint:")
print(f"  GHZ  alpha={ghz_kr_fit['alpha']:.4f},  R2={ghz_kr_fit['R2']:.4f}")
print(f"  Depth alpha={depth_kr_fit['alpha']:.4f}, R2={depth_kr_fit['R2']:.4f}")
