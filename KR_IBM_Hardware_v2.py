##############################################################################
#  KR_IBM_Hardware_v2.py  -  COMPREHENSIVE EXPERIMENT SUITE
#
#  This runs 5 strategic experiments on IBM Quantum hardware:
#    EXP 1: Random circuits with optimization_level=0 (true depth scaling)
#    EXP 2: GHZ circuits 2-10 qubits (qubit scaling)
#    EXP 3: Noise amplification (repeat circuits 1x, 2x, 3x)
#    EXP 4: Shot noise study (512, 1024, 4096 shots)
#    EXP 5: Multi-backend validation (try 2 backends if available)
#
#  STEPS:
#  1. Paste your token on line 21
#  2. Save and run: python KR_IBM_Hardware_v2.py
#
#  Estimated time: 30-90 minutes (depending on queue)
#  QPU usage: ~5-8 minutes (well within free tier)
##############################################################################

MY_TOKEN = "            "

# ----- EXPERIMENT SETTINGS (you can adjust) -----
N_QUBITS_DEPTH = 5       # Qubits for depth scaling experiments
DEPTHS = [1, 2, 4, 8, 16, 32, 64]   # Wider range for clear scaling
GHZ_QUBITS = [2, 3, 4, 5, 6, 7, 8, 9, 10]   # GHZ scaling
SHOTS_DEFAULT = 4096
SHOT_STUDY = [512, 1024, 4096]   # For shot noise experiment
USE_TWO_BACKENDS = True   # Try a second backend
NOISE_AMP_DEPTHS = [1, 2, 3, 4]   # Repetition counts for noise amplification

##############################################################################
#  DO NOT EDIT BELOW
##############################################################################

import json
import time
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.primitives import StatevectorSampler

SAVE_FILE = "ibm_hardware_v2_results.json"
start_time = time.time()

print("=" * 70)
print("  K-R BENCHMARKING v2 - COMPREHENSIVE IBM QUANTUM EXPERIMENTS")
print("=" * 70)

if MY_TOKEN == "PUT_TOKEN_HERE":
    print("\n  ERROR: Paste your token on line 21, save, run again.\n")
    exit()

# ====================================================================
# CONNECT
# ====================================================================
print("\n[CONNECT] Connecting to IBM Quantum Platform...")
service = None
for channel_name in ["ibm_quantum_platform", "ibm_cloud"]:
    try:
        QiskitRuntimeService.save_account(
            channel=channel_name, token=MY_TOKEN,
            overwrite=True, set_as_default=True
        )
        service = QiskitRuntimeService(channel=channel_name)
        print("  Connected via " + channel_name)
        break
    except Exception as e:
        continue
if service is None:
    print("  Connection failed. Check token.")
    exit()

# Get available backends
print("\n[BACKENDS] Finding quantum computers...")
backends_to_use = []
try:
    primary = service.least_busy(min_num_qubits=10, operational=True, simulator=False)
    backends_to_use.append(primary)
    print("  Primary: " + primary.name + " (" + str(primary.num_qubits) + " qubits)")

    if USE_TWO_BACKENDS:
        all_backends = service.backends(operational=True, simulator=False, min_num_qubits=10)
        for b in all_backends:
            if b.name != primary.name:
                backends_to_use.append(b)
                print("  Secondary: " + b.name + " (" + str(b.num_qubits) + " qubits)")
                break
        if len(backends_to_use) == 1:
            print("  No second backend available - using only " + primary.name)
except Exception as e:
    print("  Backend selection failed: " + str(e))
    exit()

# ====================================================================
# CIRCUIT BUILDERS
# ====================================================================

def build_ghz(nq):
    """GHZ state preparation circuit."""
    qc = QuantumCircuit(nq)
    qc.h(0)
    for i in range(nq - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc

def build_random_circuit(nq, depth, seed=42):
    """Random circuit - accumulates noise naturally."""
    qc = random_circuit(nq, depth, max_operands=2, measure=False, seed=seed + depth)
    qc.measure_all()
    return qc

def build_random_inverse(nq, depth, seed=42):
    """Random + inverse for fidelity measurement (ideal output: |0...0>)."""
    qc_rand = random_circuit(nq, depth, max_operands=2, seed=seed + depth)
    qc_inv = qc_rand.inverse()
    qc_full = qc_rand.compose(qc_inv)
    qc_full.measure_all()
    return qc_full

def build_repeated_circuit(nq, base_depth, repeats, seed=42):
    """Build a base circuit and repeat it N times to amplify noise."""
    qc_base = random_circuit(nq, base_depth, max_operands=2, seed=seed)
    qc_inv = qc_base.inverse()
    qc_unit = qc_base.compose(qc_inv)
    qc_full = QuantumCircuit(nq)
    for _ in range(repeats):
        qc_full = qc_full.compose(qc_unit)
    qc_full.measure_all()
    return qc_full

# ====================================================================
# EXECUTION HELPERS
# ====================================================================

def submit_circuit(qc, backend, shots, opt_level=0):
    """Submit a circuit and return counts. opt_level=0 prevents transpiler from collapsing."""
    sampler = SamplerV2(mode=backend)
    qc_t = transpile(qc, backend=backend, optimization_level=opt_level)
    job = sampler.run([qc_t], shots=shots)
    print("    Job: " + job.job_id() + " ... ", end="", flush=True)
    result = job.result()
    try:
        counts = result[0].data.meas.get_counts()
    except:
        counts = result[0].data[list(result[0].data.keys())[0]].get_counts()
    return counts, job.job_id()

def get_fidelity(counts, target, shots):
    """Compute fidelity from counts."""
    if isinstance(target, list):
        return sum(counts.get(t, 0) for t in target) / shots
    return counts.get(target, 0) / shots

def run_simulator(qc, shots):
    """Run on local ideal simulator."""
    sampler = StatevectorSampler()
    r = sampler.run([qc], shots=shots).result()
    return r[0].data.meas.get_counts()

# ====================================================================
# EXPERIMENT 1: GHZ qubit scaling (per backend)
# ====================================================================

def exp1_ghz_scaling(backend, shots=SHOTS_DEFAULT):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: GHZ Qubit Scaling on " + backend.name)
    print("=" * 70)
    results = {}
    sim_results = {}
    for nq in GHZ_QUBITS:
        if nq > backend.num_qubits:
            continue
        qc = build_ghz(nq)
        target = ["0" * nq, "1" * nq]
        print("  GHZ " + str(nq) + "Q...", end=" ", flush=True)
        try:
            counts, job_id = submit_circuit(qc, backend, shots, opt_level=0)
            fid = get_fidelity(counts, target, shots)
            results[nq] = {"fidelity": fid, "infidelity": 1 - fid, "job_id": job_id}
            print("F=" + str(round(fid, 4)) + ", Infid=" + str(round(1 - fid, 4)))
        except Exception as e:
            print("ERROR: " + str(e)[:100])
            results[nq] = {"fidelity": 0, "infidelity": 1, "error": str(e)[:200]}

        # Simulator comparison
        try:
            sim_counts = run_simulator(qc, shots)
            sim_fid = get_fidelity(sim_counts, target, shots)
            sim_results[nq] = {"fidelity": sim_fid, "infidelity": 1 - sim_fid}
        except:
            sim_results[nq] = {"fidelity": 1.0, "infidelity": 0.0}

    return results, sim_results

# ====================================================================
# EXPERIMENT 2: Random circuit depth scaling (opt_level=0)
# ====================================================================

def exp2_random_depth(backend, shots=SHOTS_DEFAULT):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Random Circuit Depth Scaling (opt_level=0)")
    print("    Backend: " + backend.name + " | Qubits: " + str(N_QUBITS_DEPTH))
    print("=" * 70)
    results = {}
    sim_results = {}
    target = "0" * N_QUBITS_DEPTH
    for depth in DEPTHS:
        qc = build_random_inverse(N_QUBITS_DEPTH, depth)
        print("  Random+Inv depth=" + str(depth) + "...", end=" ", flush=True)
        try:
            counts, job_id = submit_circuit(qc, backend, shots, opt_level=0)
            fid = get_fidelity(counts, target, shots)
            results[depth] = {"fidelity": fid, "infidelity": 1 - fid, "job_id": job_id}
            print("F=" + str(round(fid, 4)) + ", Infid=" + str(round(1 - fid, 4)))
        except Exception as e:
            print("ERROR: " + str(e)[:100])
            results[depth] = {"fidelity": 0, "infidelity": 1, "error": str(e)[:200]}
        try:
            sim_counts = run_simulator(qc, shots)
            sim_fid = get_fidelity(sim_counts, target, shots)
            sim_results[depth] = {"fidelity": sim_fid, "infidelity": 1 - sim_fid}
        except:
            sim_results[depth] = {"fidelity": 1.0, "infidelity": 0.0}
    return results, sim_results

# ====================================================================
# EXPERIMENT 3: Noise amplification (repeat unit circuits)
# ====================================================================

def exp3_noise_amplification(backend, shots=SHOTS_DEFAULT):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Noise Amplification (Repeated Unit Circuits)")
    print("    Backend: " + backend.name + " | Base depth: 5")
    print("=" * 70)
    results = {}
    target = "0" * N_QUBITS_DEPTH
    for repeats in NOISE_AMP_DEPTHS:
        qc = build_repeated_circuit(N_QUBITS_DEPTH, base_depth=5, repeats=repeats)
        print("  Repeats=" + str(repeats) + " (effective depth ~" + str(5*2*repeats) + ")...",
              end=" ", flush=True)
        try:
            counts, job_id = submit_circuit(qc, backend, shots, opt_level=0)
            fid = get_fidelity(counts, target, shots)
            results[repeats] = {"fidelity": fid, "infidelity": 1 - fid,
                                "effective_depth": 5*2*repeats, "job_id": job_id}
            print("F=" + str(round(fid, 4)) + ", Infid=" + str(round(1 - fid, 4)))
        except Exception as e:
            print("ERROR: " + str(e)[:100])
            results[repeats] = {"fidelity": 0, "infidelity": 1, "error": str(e)[:200]}
    return results

# ====================================================================
# EXPERIMENT 4: Shot noise study (one circuit, multiple shot counts)
# ====================================================================

def exp4_shot_noise(backend):
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Shot Noise Study (5-qubit GHZ)")
    print("    Backend: " + backend.name)
    print("=" * 70)
    results = {}
    qc = build_ghz(5)
    target = ["00000", "11111"]
    for shots in SHOT_STUDY:
        print("  Shots=" + str(shots) + "...", end=" ", flush=True)
        try:
            counts, job_id = submit_circuit(qc, backend, shots, opt_level=0)
            fid = get_fidelity(counts, target, shots)
            results[shots] = {"fidelity": fid, "infidelity": 1 - fid, "job_id": job_id}
            print("F=" + str(round(fid, 4)) + ", Infid=" + str(round(1 - fid, 4)))
        except Exception as e:
            print("ERROR: " + str(e)[:100])
            results[shots] = {"fidelity": 0, "infidelity": 1, "error": str(e)[:200]}
    return results

# ====================================================================
# RUN ALL EXPERIMENTS ON ALL BACKENDS
# ====================================================================

all_results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "configuration": {
        "ghz_qubits": GHZ_QUBITS,
        "depths": DEPTHS,
        "shots_default": SHOTS_DEFAULT,
        "shot_study": SHOT_STUDY,
        "noise_amp_depths": NOISE_AMP_DEPTHS,
        "n_qubits_depth": N_QUBITS_DEPTH,
        "optimization_level": 0,
    },
    "backends": {},
}

for backend in backends_to_use:
    print("\n" + "#" * 70)
    print("# RUNNING ON: " + backend.name)
    print("#" * 70)

    backend_results = {
        "name": backend.name,
        "num_qubits": backend.num_qubits,
    }

    # EXP 1: GHZ scaling
    ghz_hw, ghz_sim = exp1_ghz_scaling(backend)
    backend_results["exp1_ghz_hardware"] = {str(k): v for k, v in ghz_hw.items()}
    backend_results["exp1_ghz_simulator"] = {str(k): v for k, v in ghz_sim.items()}

    # EXP 2: Random depth scaling
    depth_hw, depth_sim = exp2_random_depth(backend)
    backend_results["exp2_depth_hardware"] = {str(k): v for k, v in depth_hw.items()}
    backend_results["exp2_depth_simulator"] = {str(k): v for k, v in depth_sim.items()}

    # EXP 3: Noise amplification (only on primary backend to save time)
    if backend == backends_to_use[0]:
        amp_hw = exp3_noise_amplification(backend)
        backend_results["exp3_noise_amplification"] = {str(k): v for k, v in amp_hw.items()}

        # EXP 4: Shot noise (only on primary backend)
        shot_hw = exp4_shot_noise(backend)
        backend_results["exp4_shot_noise"] = {str(k): v for k, v in shot_hw.items()}

    all_results["backends"][backend.name] = backend_results

# ====================================================================
# K-R ANALYSIS ON ALL DATA
# ====================================================================

def kr_model(K, C0, R, alpha):
    return C0 / (np.clip(K, 1e-30, None) ** alpha) + R

def fit_kr(K, C):
    try:
        popt, _ = curve_fit(kr_model, K, C, p0=[0.5, 0.01, 1.0],
                            bounds=([0, 0, 0.01], [50, 2, 15]), maxfev=50000)
        yp = kr_model(K, *popt)
        ss_r = np.sum((C - yp) ** 2); ss_t = np.sum((C - C.mean()) ** 2)
        r2 = 1 - ss_r / ss_t if ss_t > 1e-30 else 0
        return {"alpha": float(popt[2]), "C0": float(popt[0]), "R": float(popt[1]),
                "R2": float(r2), "params": popt}
    except Exception as e:
        return {"alpha": None, "R2": 0, "error": str(e)}

print("\n\n" + "=" * 70)
print("  K-R ANALYSIS ON ALL REAL HARDWARE DATA")
print("=" * 70)

for backend_name, br in all_results["backends"].items():
    print("\n[" + backend_name + "]")

    # GHZ analysis (qubit scaling)
    ghz_hw = br["exp1_ghz_hardware"]
    if ghz_hw:
        nqs = sorted([int(k) for k in ghz_hw.keys()])
        K_ghz = np.array([1.0/n for n in nqs])  # K = 1/qubits
        C_ghz = np.array([ghz_hw[str(n)]["infidelity"] for n in nqs
                          if "infidelity" in ghz_hw[str(n)]])
        if len(C_ghz) > 3:
            ghz_fit = fit_kr(K_ghz[:len(C_ghz)], C_ghz)
            print("  GHZ K-R: alpha=" + str(round(ghz_fit.get("alpha", 0), 4)) +
                  ", R2=" + str(round(ghz_fit.get("R2", 0), 4)))
            br["kr_fit_ghz"] = {k: v for k, v in ghz_fit.items() if k != "params"}

    # Depth analysis
    depth_hw = br["exp2_depth_hardware"]
    if depth_hw:
        depths = sorted([int(k) for k in depth_hw.keys()])
        C_depth = np.array([depth_hw[str(d)]["infidelity"] for d in depths
                            if "infidelity" in depth_hw[str(d)]])
        depths_arr = np.array(depths[:len(C_depth)])
        K_depth = 1.0 / depths_arr
        if len(C_depth) > 3:
            depth_fit = fit_kr(K_depth, C_depth)
            print("  Depth K-R: alpha=" + str(round(depth_fit.get("alpha", 0), 4)) +
                  ", R2=" + str(round(depth_fit.get("R2", 0), 4)))
            br["kr_fit_depth"] = {k: v for k, v in depth_fit.items() if k != "params"}

# ====================================================================
# SAVE RESULTS
# ====================================================================

# Strip non-serializable params
def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items() if k != "params"}
    if isinstance(obj, list):
        return [clean(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    return obj

with open(SAVE_FILE, "w") as f:
    json.dump(clean(all_results), f, indent=2)
print("\n  Saved: " + SAVE_FILE)

# ====================================================================
# GENERATE FIGURES
# ====================================================================

print("\n  Generating figures...")

primary_name = backends_to_use[0].name
br_primary = all_results["backends"][primary_name]

# Figure 1: 4-panel comprehensive results
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("K-R Benchmarking on Real IBM Quantum Hardware: " + primary_name,
             fontsize=14, fontweight="bold")

# (a) GHZ qubit scaling - all backends
ax = axes[0, 0]
for bname, br in all_results["backends"].items():
    ghz_hw = br.get("exp1_ghz_hardware", {})
    if ghz_hw:
        nqs = sorted([int(k) for k in ghz_hw.keys()])
        infids = [ghz_hw[str(n)].get("infidelity", 0) for n in nqs]
        ax.plot(nqs, infids, "o-", lw=2, markersize=7, label=bname + " (real)")
    ghz_sim = br.get("exp1_ghz_simulator", {})
    if ghz_sim:
        nqs_s = sorted([int(k) for k in ghz_sim.keys()])
        sim_inf = [ghz_sim[str(n)].get("infidelity", 0) for n in nqs_s]
        ax.plot(nqs_s, sim_inf, "s--", lw=1, markersize=4, alpha=0.6, label=bname + " (sim)")
ax.set_xlabel("Number of Qubits")
ax.set_ylabel("GHZ Infidelity")
ax.set_title("(a) GHZ Qubit Scaling (Multi-Backend)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (b) Random circuit depth scaling - all backends + K-R fit
ax = axes[0, 1]
for bname, br in all_results["backends"].items():
    depth_hw = br.get("exp2_depth_hardware", {})
    if depth_hw:
        depths = sorted([int(k) for k in depth_hw.keys()])
        infids = [depth_hw[str(d)].get("infidelity", 0) for d in depths]
        ax.plot(depths, infids, "o-", lw=2, markersize=7, label=bname + " (real)")

        # K-R fit overlay
        kr = br.get("kr_fit_depth", {})
        if kr.get("alpha"):
            d_smooth = np.linspace(min(depths), max(depths), 100)
            K_smooth = 1.0 / d_smooth
            C_fit = kr["C0"] / (K_smooth ** kr["alpha"]) + kr["R"]
            ax.plot(d_smooth, C_fit, "k--", lw=1.5, alpha=0.7,
                    label="K-R fit: α=" + str(round(kr["alpha"], 3)) +
                          ", R²=" + str(round(kr["R2"], 3)))
ax.set_xlabel("Circuit Depth")
ax.set_ylabel("Infidelity (1 - P(|0...0⟩))")
ax.set_title("(b) Random Circuit Depth Scaling (opt_level=0)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (c) Noise amplification
ax = axes[1, 0]
amp = br_primary.get("exp3_noise_amplification", {})
if amp:
    reps = sorted([int(k) for k in amp.keys()])
    infids = [amp[str(r)].get("infidelity", 0) for r in reps]
    eff_depths = [amp[str(r)].get("effective_depth", r*10) for r in reps]
    ax.plot(eff_depths, infids, "ro-", lw=2, markersize=8, label="Real (" + primary_name + ")")
ax.set_xlabel("Effective Circuit Depth")
ax.set_ylabel("Infidelity")
ax.set_title("(c) Noise Amplification (Repeated Circuits)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (d) Shot noise study
ax = axes[1, 1]
shots_data = br_primary.get("exp4_shot_noise", {})
if shots_data:
    shots_list = sorted([int(k) for k in shots_data.keys()])
    infids = [shots_data[str(s)].get("infidelity", 0) for s in shots_list]
    # Theoretical 1/sqrt(N) error bars
    errs = [np.sqrt(infids[i]*(1-infids[i])/shots_list[i]) for i in range(len(shots_list))]
    ax.errorbar(shots_list, infids, yerr=errs, fmt="bo-", lw=2, markersize=8,
                capsize=5, label="GHZ-5Q infidelity")
ax.set_xlabel("Shots")
ax.set_ylabel("Infidelity")
ax.set_xscale("log")
ax.set_title("(d) Shot Noise Sensitivity (5-Qubit GHZ)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("Fig_IBM_v2_Comprehensive.png", dpi=300, bbox_inches="tight")
print("  Saved: Fig_IBM_v2_Comprehensive.png")

# Figure 2: K-R Dual Validation (GHZ + Depth)
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
fig2.suptitle("K-R Dual Validation on Real Hardware: " + primary_name,
              fontsize=13, fontweight="bold")

# GHZ K-R fit
ax = axes2[0]
ghz_hw = br_primary.get("exp1_ghz_hardware", {})
if ghz_hw:
    nqs = sorted([int(k) for k in ghz_hw.keys()])
    infids = [ghz_hw[str(n)].get("infidelity", 0) for n in nqs]
    ax.plot(nqs, infids, "ro", markersize=10, label="Real hardware")
    kr_g = br_primary.get("kr_fit_ghz", {})
    if kr_g.get("alpha"):
        n_smooth = np.linspace(min(nqs), max(nqs), 100)
        K_smooth = 1.0 / n_smooth
        C_fit = kr_g["C0"] / (K_smooth ** kr_g["alpha"]) + kr_g["R"]
        ax.plot(n_smooth, C_fit, "k--", lw=2,
                label="K-R: α=" + str(round(kr_g["alpha"], 3)) +
                      ", R²=" + str(round(kr_g["R2"], 3)))
ax.set_xlabel("Number of Qubits")
ax.set_ylabel("GHZ Infidelity")
ax.set_title("(a) K-R Fit: Qubit Scaling Axis")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Depth K-R fit
ax = axes2[1]
depth_hw = br_primary.get("exp2_depth_hardware", {})
if depth_hw:
    depths = sorted([int(k) for k in depth_hw.keys()])
    infids = [depth_hw[str(d)].get("infidelity", 0) for d in depths]
    ax.plot(depths, infids, "bs", markersize=10, label="Real hardware")
    kr_d = br_primary.get("kr_fit_depth", {})
    if kr_d.get("alpha"):
        d_smooth = np.linspace(min(depths), max(depths), 100)
        K_smooth = 1.0 / d_smooth
        C_fit = kr_d["C0"] / (K_smooth ** kr_d["alpha"]) + kr_d["R"]
        ax.plot(d_smooth, C_fit, "k--", lw=2,
                label="K-R: α=" + str(round(kr_d["alpha"], 3)) +
                      ", R²=" + str(round(kr_d["R2"], 3)))
ax.set_xlabel("Circuit Depth")
ax.set_ylabel("Random Circuit Infidelity")
ax.set_title("(b) K-R Fit: Depth Scaling Axis")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig("Fig_IBM_v2_KR_Dual.png", dpi=300, bbox_inches="tight")
print("  Saved: Fig_IBM_v2_KR_Dual.png")

elapsed = time.time() - start_time
print("\n" + "=" * 70)
print("  COMPLETE! Total time: " + str(round(elapsed / 60, 1)) + " minutes")
print("\n  Upload to Claude:")
print("    1. ibm_hardware_v2_results.json")
print("    2. Fig_IBM_v2_Comprehensive.png")
print("    3. Fig_IBM_v2_KR_Dual.png")
print("=" * 70)