"""
K-R Real Hardware Experiments — Paper Strengthening
Author: Ramakrishna Pasupuleti
Date: July 2026

Six targeted real-hardware experiments:

EXP-H1: Temporal Drift Detection (5 repeated runs over time)
         Validates Calibration Trigger Metric (CTM) on real hardware
         
EXP-H2: Cross-Architecture (ibm_sherbrooke, Eagle r3 127Q)
         Shows alpha captures architecture differences

EXP-H3: Noise Amplification on ibm_marrakesh
         Completes three-device amplification comparison

EXP-H4: Shot Noise Stability on ibm_marrakesh
         Confirms CV < 3% across devices

EXP-H5: Depth Scaling n=10 on ibm_marrakesh
         Tests alpha_depth vs system size

EXP-H6: Large GHZ on ibm_sherbrooke (if available)
         Cross-architecture GHZ fingerprint
"""

import json
import time
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IBM_TOKEN    = "0Y4TTLJhI5euf-S0fmsCeLPVT0Vgr4H-rZZz0osJgfoG"
SHOTS        = 4096
OPT_LEVEL    = 0

# Primary backend (already fingerprinted)
PRIMARY_BACKEND   = "ibm_marrakesh"

# Secondary backend (different architecture)
SECONDARY_BACKEND = "ibm_sherbrooke"   # Eagle r3, 127Q

# Experiment switches — set False to skip
RUN_H1_TEMPORAL    = True   # 5 repeated GHZ runs
RUN_H2_SHERBROOKE  = True   # Cross-architecture
RUN_H3_AMPLIF      = True   # Noise amplification marrakesh
RUN_H4_SHOTNOISE   = True   # Shot noise stability
RUN_H5_DEPTH_N10   = True   # Depth scaling n=10
RUN_H6_GHZ_LARGE   = True   # Large GHZ cross-arch

# ─────────────────────────────────────────────
# CONNECT
# ─────────────────────────────────────────────
print("Connecting to IBM Quantum...")
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=IBM_TOKEN
)

# Load primary backend
backend_primary = service.backend(PRIMARY_BACKEND)
print(f"Primary:   {backend_primary.name} ({backend_primary.num_qubits}Q)")

# Try secondary backend
backend_secondary = None
try:
    backend_secondary = service.backend(SECONDARY_BACKEND)
    print(f"Secondary: {backend_secondary.name} ({backend_secondary.num_qubits}Q)")
except Exception as e:
    print(f"Secondary backend {SECONDARY_BACKEND} not available: {e}")
    print("Will skip cross-architecture experiments")
    RUN_H2_SHERBROOKE = False
    RUN_H6_GHZ_LARGE  = False

print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ─────────────────────────────────────────────
# CIRCUIT BUILDERS
# ─────────────────────────────────────────────
def build_ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc

def build_mirror(n, depth):
    import random
    random.seed(42 + depth)
    qc = QuantumCircuit(n)
    seq = []
    for _ in range(depth):
        layer = [(random.choice(['h','s','x','y','z']), q) for q in range(n)]
        seq.append(layer)
    for layer in seq:
        for gate, q in layer:
            getattr(qc, gate)(q)
    for layer in reversed(seq):
        for gate, q in layer:
            qc.sdg(q) if gate == 's' else getattr(qc, gate)(q)
    qc.measure_all()
    return qc

def build_noise_amplified(n, depth, repetitions):
    """Noise amplification via gate folding"""
    qc = QuantumCircuit(n)
    for _ in range(repetitions):
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n - 2, -1, -1):
            qc.cx(i, i + 1)
        qc.h(0)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc

# ─────────────────────────────────────────────
# FIDELITY HELPERS
# ─────────────────────────────────────────────
def ghz_fidelity(counts, n):
    total = sum(counts.values())
    p0 = counts.get('0'*n, 0) / total
    p1 = counts.get('1'*n, 0) / total
    return p0+p1, 1.0-(p0+p1)

def mirror_fidelity(counts, n):
    total = sum(counts.values())
    f = counts.get('0'*n, 0) / total
    return f, 1.0-f

# ─────────────────────────────────────────────
# JOB RUNNER WITH RETRY
# ─────────────────────────────────────────────
def run_job(qc, backend, label, retries=5):
    qc_t = transpile(qc, backend=backend, optimization_level=OPT_LEVEL)
    sampler = Sampler(mode=backend)
    for attempt in range(retries):
        try:
            print(f"  Submitting {label} (attempt {attempt+1})...")
            job = sampler.run([qc_t], shots=SHOTS)
            jid = job.job_id()
            print(f"  Job ID: {jid} — waiting...")
            result = job.result()
            counts = result[0].data.meas.get_counts()
            print(f"  Done.")
            return counts, jid
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < retries-1:
                print(f"  Retrying in 30s...")
                time.sleep(30)
    return None, None

# ─────────────────────────────────────────────
# K-R FITTING
# ─────────────────────────────────────────────
def kr_model(K, C0, alpha, R):
    return C0 / (K**alpha) + R

def fit_kr(x_vals, inf_vals):
    x = np.array(x_vals, dtype=float)
    y = np.array(inf_vals, dtype=float)
    K = 1.0/x
    try:
        popt, _ = curve_fit(kr_model, K, y,
            p0=[0.1, 0.5, 0.01],
            bounds=([0,0.001,0],[50,15,1]),
            method='trf', maxfev=20000)
        C0, alpha, R = popt
        y_pred = kr_model(K, *popt)
        ss_res = np.sum((y-y_pred)**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        R2 = 1.0-ss_res/ss_tot if ss_tot>0 else 0.0
        return {"alpha":float(alpha),"C0":float(C0),
                "R":float(R),"R2":float(R2),"ok":True}
    except Exception as e:
        return {"alpha":None,"R2":None,"ok":False,"error":str(e)}

# ─────────────────────────────────────────────
# RESULTS STORAGE
# ─────────────────────────────────────────────
results = {
    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "shots": SHOTS,
    "backends": {
        "primary": PRIMARY_BACKEND,
        "secondary": SECONDARY_BACKEND
    }
}

def save_partial():
    with open("kr_hardware_experiments.json","w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  [Partial results saved]")

# ═══════════════════════════════════════════════════════════
# EXP-H1: TEMPORAL DRIFT DETECTION
# Run same GHZ-5 circuit 5 times spaced ~2 minutes apart
# Measures alpha stability and drift on real hardware
# ═══════════════════════════════════════════════════════════
if RUN_H1_TEMPORAL:
    print("\n" + "="*60)
    print("EXP-H1: TEMPORAL DRIFT DETECTION")
    print("5 repeated GHZ-5 runs on ibm_marrakesh")
    print("="*60)

    h1_runs = []
    qc_ghz5 = build_ghz(5)
    N_RUNS = 5

    for run_idx in range(N_RUNS):
        t_start = datetime.now().strftime('%H:%M:%S')
        counts, jid = run_job(
            qc_ghz5, backend_primary,
            f"H1 GHZ-5 Run {run_idx+1}/{N_RUNS}"
        )
        if counts:
            fid, inf = ghz_fidelity(counts, 5)
            h1_runs.append({
                "run": run_idx+1,
                "timestamp": t_start,
                "fidelity": float(fid),
                "infidelity": float(inf),
                "job_id": jid
            })
            print(f"  Run {run_idx+1}: fidelity={fid:.4f}, "
                  f"infidelity={inf:.4f}")

    if len(h1_runs) >= 3:
        inf_vals = [r["infidelity"] for r in h1_runs]
        mean_inf = float(np.mean(inf_vals))
        std_inf  = float(np.std(inf_vals))
        cv_inf   = float(std_inf/mean_inf*100) if mean_inf > 0 else 0.0

        # Fit KR to the run-index as x-axis
        run_indices = list(range(1, len(h1_runs)+1))
        kr_fit = fit_kr(run_indices, inf_vals)

        print(f"\n  Temporal stability:")
        print(f"  Mean infidelity = {mean_inf:.4f} ± {std_inf:.4f}")
        print(f"  CV = {cv_inf:.2f}%")
        print(f"  Drift detected: {cv_inf > 5.0}")

        results["h1_temporal_drift"] = {
            "description": "5 repeated GHZ-5 runs for temporal stability",
            "backend": PRIMARY_BACKEND,
            "circuit": "GHZ n=5",
            "runs": h1_runs,
            "mean_infidelity": mean_inf,
            "std_infidelity": std_inf,
            "cv_percent": cv_inf,
            "drift_detected": cv_inf > 5.0,
            "stable": cv_inf < 5.0
        }
        save_partial()

# ═══════════════════════════════════════════════════════════
# EXP-H2: CROSS-ARCHITECTURE (ibm_sherbrooke)
# GHZ scaling on Eagle r3 vs Heron r2
# ═══════════════════════════════════════════════════════════
if RUN_H2_SHERBROOKE and backend_secondary:
    print("\n" + "="*60)
    print("EXP-H2: CROSS-ARCHITECTURE")
    print(f"GHZ scaling on {SECONDARY_BACKEND} (Eagle r3)")
    print("="*60)

    ghz_n_list = [2, 5, 10, 15, 20]
    h2_results = {}

    for n in ghz_n_list:
        qc = build_ghz(n)
        counts, jid = run_job(qc, backend_secondary, f"H2 GHZ n={n}")
        if counts:
            fid, inf = ghz_fidelity(counts, n)
            h2_results[str(n)] = {
                "n_qubits": n,
                "fidelity": float(fid),
                "infidelity": float(inf),
                "job_id": jid
            }
            print(f"  n={n}: fidelity={fid:.4f}, infidelity={inf:.4f}")

    if len(h2_results) >= 3:
        n_vals  = sorted([int(k) for k in h2_results])
        inf_vals = [h2_results[str(n)]["infidelity"] for n in n_vals]
        kr_fit  = fit_kr(n_vals, inf_vals)
        print(f"\n  {SECONDARY_BACKEND} GHZ K-R fit:")
        print(f"  alpha={kr_fit['alpha']:.4f}, R2={kr_fit['R2']:.4f}")

        results["h2_cross_architecture"] = {
            "description": "GHZ scaling on Eagle r3 for cross-architecture comparison",
            "backend": SECONDARY_BACKEND,
            "architecture": "Eagle r3 127Q",
            "data": h2_results,
            "kr_fit": kr_fit,
            "comparison_note": (
                f"Heron r2 (marrakesh) alpha_GHZ=1.4055 vs "
                f"Eagle r3 ({SECONDARY_BACKEND}) alpha_GHZ="
                f"{kr_fit['alpha']:.4f if kr_fit['ok'] else 'N/A'}"
            )
        }
        save_partial()

# ═══════════════════════════════════════════════════════════
# EXP-H3: NOISE AMPLIFICATION on ibm_marrakesh
# Completes three-device amplification comparison
# You have kingston (alpha=1.190, R2=1.000)
# Now get marrakesh amplification alpha
# ═══════════════════════════════════════════════════════════
if RUN_H3_AMPLIF:
    print("\n" + "="*60)
    print("EXP-H3: NOISE AMPLIFICATION — ibm_marrakesh")
    print("Repetitions: 1, 2, 3, 4")
    print("="*60)

    rep_list = [1, 2, 3, 4]
    h3_results = {}
    N_AMP = 5

    for rep in rep_list:
        qc = build_noise_amplified(N_AMP, 10, rep)
        counts, jid = run_job(
            qc, backend_primary,
            f"H3 Amplif rep={rep}"
        )
        if counts:
            fid, inf = ghz_fidelity(counts, N_AMP)
            h3_results[str(rep)] = {
                "repetitions": rep,
                "effective_depth": 10 * (2*rep + 1),
                "fidelity": float(fid),
                "infidelity": float(inf),
                "job_id": jid
            }
            print(f"  rep={rep}: fidelity={fid:.4f}, "
                  f"infidelity={inf:.4f}")

    if len(h3_results) >= 3:
        rep_vals = sorted([int(k) for k in h3_results])
        inf_vals = [h3_results[str(r)]["infidelity"] for r in rep_vals]
        kr_fit   = fit_kr(rep_vals, inf_vals)
        print(f"\n  Amplification K-R fit:")
        print(f"  alpha={kr_fit['alpha']:.4f}, R2={kr_fit['R2']:.4f}")
        print(f"  Compare: ibm_kingston alpha=1.190, R2=1.000")

        results["h3_noise_amplification"] = {
            "description": "Noise amplification via gate folding on ibm_marrakesh",
            "backend": PRIMARY_BACKEND,
            "n_qubits": N_AMP,
            "data": h3_results,
            "kr_fit": kr_fit,
            "kingston_reference": {
                "alpha": 1.1904, "R2": 1.0000
            }
        }
        save_partial()

# ═══════════════════════════════════════════════════════════
# EXP-H4: SHOT NOISE STABILITY on ibm_marrakesh
# Confirms CV < 3% across shot counts
# ═══════════════════════════════════════════════════════════
if RUN_H4_SHOTNOISE:
    print("\n" + "="*60)
    print("EXP-H4: SHOT NOISE STABILITY — ibm_marrakesh")
    print("Shot counts: 512, 1024, 4096")
    print("="*60)

    shot_list = [512, 1024, 4096]
    h4_results = {}
    qc_5q = build_ghz(5)
    qc_t  = transpile(qc_5q, backend=backend_primary,
                      optimization_level=OPT_LEVEL)

    for shots in shot_list:
        sampler = Sampler(mode=backend_primary)
        try:
            print(f"  Submitting shots={shots}...")
            job    = sampler.run([qc_t], shots=shots)
            jid    = job.job_id()
            print(f"  Job ID: {jid} — waiting...")
            result = job.result()
            counts = result[0].data.meas.get_counts()
            fid, inf = ghz_fidelity(counts, 5)
            h4_results[str(shots)] = {
                "shots": shots,
                "fidelity": float(fid),
                "infidelity": float(inf),
                "job_id": jid
            }
            print(f"  shots={shots}: fidelity={fid:.4f}, "
                  f"infidelity={inf:.4f}")
        except Exception as e:
            print(f"  Error for shots={shots}: {e}")

    if len(h4_results) >= 2:
        inf_vals = [h4_results[str(s)]["infidelity"]
                    for s in shot_list if str(s) in h4_results]
        mean_inf = float(np.mean(inf_vals))
        std_inf  = float(np.std(inf_vals))
        cv       = float(std_inf/mean_inf*100) if mean_inf > 0 else 0

        print(f"\n  Shot noise CV = {cv:.2f}%")
        print(f"  Stable (CV<3%): {cv < 3.0}")

        results["h4_shot_noise"] = {
            "description": "Shot noise stability on ibm_marrakesh GHZ-5",
            "backend": PRIMARY_BACKEND,
            "data": h4_results,
            "mean_infidelity": mean_inf,
            "cv_percent": cv,
            "stable": cv < 3.0
        }
        save_partial()

# ═══════════════════════════════════════════════════════════
# EXP-H5: DEPTH SCALING n=10 on ibm_marrakesh
# Tests whether alpha_depth changes with system size
# Previous: n=5 gave alpha=1.493
# ═══════════════════════════════════════════════════════════
if RUN_H5_DEPTH_N10:
    print("\n" + "="*60)
    print("EXP-H5: DEPTH SCALING n=10 — ibm_marrakesh")
    print("Depths: 1,2,4,8,16,32,64")
    print("="*60)

    depth_list = [1, 2, 4, 8, 16, 32, 64]
    h5_results = {}
    N_DEPTH = 10

    for d in depth_list:
        qc = build_mirror(N_DEPTH, d)
        counts, jid = run_job(
            qc, backend_primary,
            f"H5 Depth n=10 d={d}"
        )
        if counts:
            fid, inf = mirror_fidelity(counts, N_DEPTH)
            h5_results[str(d)] = {
                "depth": d,
                "fidelity": float(fid),
                "infidelity": float(inf),
                "job_id": jid
            }
            print(f"  d={d}: fidelity={fid:.4f}, "
                  f"infidelity={inf:.4f}")
        save_partial()

    if len(h5_results) >= 4:
        d_vals   = sorted([int(k) for k in h5_results])
        inf_vals = [h5_results[str(d)]["infidelity"] for d in d_vals]
        kr_fit   = fit_kr(d_vals, inf_vals)
        print(f"\n  Depth n=10 K-R fit:")
        print(f"  alpha={kr_fit['alpha']:.4f}, R2={kr_fit['R2']:.4f}")
        print(f"  Compare n=5: alpha=1.493, R2=0.900")

        results["h5_depth_n10"] = {
            "description": "Depth scaling with n=10 qubits on ibm_marrakesh",
            "backend": PRIMARY_BACKEND,
            "n_qubits": N_DEPTH,
            "depth_list": depth_list,
            "data": h5_results,
            "kr_fit": kr_fit,
            "n5_reference": {
                "alpha": 1.4926, "R2": 0.8999
            }
        }
        save_partial()

# ═══════════════════════════════════════════════════════════
# EXP-H6: LARGE GHZ on ibm_sherbrooke
# Cross-architecture GHZ fingerprint
# ═══════════════════════════════════════════════════════════
if RUN_H6_GHZ_LARGE and backend_secondary:
    print("\n" + "="*60)
    print("EXP-H6: LARGE GHZ — ibm_sherbrooke")
    print("GHZ n=2,5,10,15,20,25,30")
    print("="*60)

    ghz_list = [2, 5, 10, 15, 20, 25, 30]
    h6_results = {}

    for n in ghz_list:
        qc = build_ghz(n)
        counts, jid = run_job(
            qc, backend_secondary,
            f"H6 GHZ n={n}"
        )
        if counts:
            fid, inf = ghz_fidelity(counts, n)
            h6_results[str(n)] = {
                "n_qubits": n,
                "fidelity": float(fid),
                "infidelity": float(inf),
                "job_id": jid
            }
            print(f"  n={n}: fidelity={fid:.4f}, "
                  f"infidelity={inf:.4f}")
        save_partial()

    if len(h6_results) >= 4:
        n_vals   = sorted([int(k) for k in h6_results])
        inf_vals = [h6_results[str(n)]["infidelity"] for n in n_vals]
        kr_fit   = fit_kr(n_vals, inf_vals)
        print(f"\n  {SECONDARY_BACKEND} GHZ K-R fit:")
        print(f"  alpha={kr_fit['alpha']:.4f}, R2={kr_fit['R2']:.4f}")

        results["h6_ghz_sherbrooke"] = {
            "description": "GHZ scaling to n=30 on ibm_sherbrooke Eagle r3",
            "backend": SECONDARY_BACKEND,
            "architecture": "Eagle r3",
            "data": h6_results,
            "kr_fit": kr_fit
        }
        save_partial()

# ─────────────────────────────────────────────
# FINAL SAVE AND SUMMARY
# ─────────────────────────────────────────────
results["completion_timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with open("kr_hardware_experiments.json","w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "="*60)
print("ALL HARDWARE EXPERIMENTS COMPLETE")
print("="*60)
print(f"Saved to: kr_hardware_experiments.json")
print(f"\nCompleted experiments:")
for key in results:
    if key.startswith('h') and isinstance(results[key], dict):
        print(f"  {key}: {results[key].get('description','')}")

print("\nUpload kr_hardware_experiments.json for manuscript update.")
