##############################################################################
#  KR_IBM_8_Regimes.py
#
#  Tests K-R noise classifier on REAL IBM Quantum hardware across regimes
#  that you can directly probe (cannot tune T1/T2/crosstalk - hardware
#  fixes those - so we test what's controllable on real machines).
#
#  6 EXPERIMENTS DESIGNED FOR REAL HARDWARE:
#    EXP 1: Coherent over-rotation circuits (predicted alpha ~ 2)
#    EXP 2: Random Clifford depth scaling (true noise accumulation)
#    EXP 3: GHZ qubit scaling 2-10 qubits
#    EXP 4: Noise amplification (1x, 2x, 3x, 4x repeats)
#    EXP 5: Echo experiment (X-X identity, tests gate quality vs depth)
#    EXP 6: Long-depth random circuits (saturation analysis)
#
#  STEPS:
#  1. Paste your token on line 21
#  2. Save and run: python KR_IBM_8_Regimes.py
#
#  Estimated time: 30-60 minutes (depends on queue)
#  QPU usage: ~5-7 minutes (within free tier 10 min/month)
##############################################################################

MY_TOKEN = "            "

# ----- EXPERIMENT SETTINGS -----
SHOTS = 4096
N_QUBITS_DEPTH = 5           # Qubits for depth experiments
DEPTHS = [1, 2, 4, 8, 16, 32, 64]
GHZ_QUBITS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
COHERENT_DEPTHS = [1, 4, 8, 16, 32, 64, 128]
COHERENT_EPS = 0.05          # Over-rotation per gate (radians)
NOISE_AMP_REPEATS = [1, 2, 3, 4]
ECHO_DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128]
LONG_DEPTHS = [10, 25, 50, 100, 200]
USE_TWO_BACKENDS = True

##############################################################################
#  DO NOT EDIT BELOW
##############################################################################

import json
import time
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.primitives import StatevectorSampler

SAVE_FILE = "ibm_8_regimes_results.json"
start_time = time.time()

print("=" * 72)
print("  K-R 8-REGIME VALIDATION ON REAL IBM QUANTUM HARDWARE")
print("=" * 72)

if MY_TOKEN == "PUT_TOKEN_HERE":
    print("\n  ERROR: Paste your token on line 21, save, run again.\n")
    exit()

# ====================================================================
# CONNECT
# ====================================================================
print("\n[CONNECT] Connecting to IBM Quantum Platform...")
service = None
for channel in ["ibm_quantum_platform", "ibm_cloud"]:
    try:
        QiskitRuntimeService.save_account(
            channel=channel, token=MY_TOKEN,
            overwrite=True, set_as_default=True
        )
        service = QiskitRuntimeService(channel=channel)
        print("  Connected via " + channel)
        break
    except Exception as e:
        continue

if service is None:
    print("  Connection failed. Check token.")
    exit()

# ====================================================================
# SELECT BACKENDS
# ====================================================================
print("\n[BACKENDS] Selecting quantum computers...")
backends_to_use = []
try:
    primary = service.least_busy(min_num_qubits=10, operational=True, simulator=False)
    backends_to_use.append(primary)
    print("  Primary: " + primary.name + " (" + str(primary.num_qubits) + " qubits)")

    if USE_TWO_BACKENDS:
        all_b = service.backends(operational=True, simulator=False, min_num_qubits=10)
        for b in all_b:
            if b.name != primary.name:
                backends_to_use.append(b)
                print("  Secondary: " + b.name + " (" + str(b.num_qubits) + " qubits)")
                break
except Exception as e:
    print("  Backend error: " + str(e))
    exit()

# ====================================================================
# CIRCUIT BUILDERS
# ====================================================================

def build_coherent_drift(nq, depth, eps):
    """RX(eps) drift circuit. Errors accumulate coherently. Predicted alpha ~ 2."""
    qc = QuantumCircuit(nq)
    for _ in range(depth):
        for q in range(nq):
            qc.rx(eps, q)
    qc.measure_all()
    return qc

def build_clifford_depth(nq, depth):
    """Deterministic Clifford layers + inverse. Ideal output: |0...0>."""
    qc = QuantumCircuit(nq)
    for layer in range(depth):
        for q in range(nq):
            qc.h(q)
        for q in range(0, nq-1, 2):
            qc.cx(q, q+1)
        for q in range(nq):
            qc.s(q)
        if nq > 2:
            for q in range(1, nq-1, 2):
                qc.cx(q, q+1)
    qc_inv = qc.inverse()
    qc_full = qc.compose(qc_inv)
    qc_full.measure_all()
    return qc_full

def build_ghz(nq):
    """GHZ state: 50% |00..0> + 50% |11..1>."""
    qc = QuantumCircuit(nq)
    qc.h(0)
    for i in range(nq - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc

def build_repeated_unit(nq, base_depth, repeats):
    """Repeat unit Clifford+inverse circuit N times for noise amplification."""
    qc_unit = build_clifford_depth(nq, base_depth)
    qc_unit.remove_final_measurements()
    qc_full = QuantumCircuit(nq)
    for _ in range(repeats):
        qc_full = qc_full.compose(qc_unit)
    qc_full.measure_all()
    return qc_full

def build_echo_circuit(nq, depth):
    """Echo: X gate applied 2*depth times. Ideal output: |00..0>."""
    qc = QuantumCircuit(nq)
    for _ in range(2 * depth):
        for q in range(nq):
            qc.x(q)
    qc.measure_all()
    return qc

def build_random_long(nq, depth):
    """Long random circuit + inverse for saturation analysis."""
    qc = random_circuit(nq, depth, max_operands=2, seed=42 + depth)
    qc_inv = qc.inverse()
    qc_full = qc.compose(qc_inv)
    qc_full.measure_all()
    return qc_full

# ====================================================================
# EXECUTION HELPER
# ====================================================================

def submit(qc, backend, shots=SHOTS, opt=0):
    sampler = SamplerV2(mode=backend)
    qc_t = transpile(qc, backend=backend, optimization_level=opt)
    job = sampler.run([qc_t], shots=shots)
    print("    Job: " + job.job_id() + "...", end=" ", flush=True)
    result = job.result()
    try:
        counts = result[0].data.meas.get_counts()
    except:
        counts = result[0].data[list(result[0].data.keys())[0]].get_counts()
    return counts, job.job_id()

def get_fid(counts, target, shots):
    if isinstance(target, list):
        return sum(counts.get(t, 0) for t in target) / shots
    return counts.get(target, 0) / shots

# ====================================================================
# K-R ANALYSIS HELPERS
# ====================================================================

def kr(K, C0, R, alpha):
    return C0 / (np.clip(K, 1e-30, None) ** alpha) + R

def fit_kr(K, C):
    try:
        popt, _ = curve_fit(kr, K, C, p0=[0.5, 0.01, 1.0],
                            bounds=([0, 0, 0.01], [50, 2, 15]), maxfev=80000)
        yp = kr(K, *popt)
        ssr = np.sum((C-yp)**2)
        sst = np.sum((C-C.mean())**2)
        r2 = 1 - ssr/sst if sst > 1e-30 else 0
        return {"alpha": float(popt[2]), "C0": float(popt[0]),
                "R": float(popt[1]), "R2": float(r2), "ok": True}
    except Exception as e:
        return {"alpha": None, "R2": 0, "ok": False, "error": str(e)}

# ====================================================================
# RUN EXPERIMENTS PER BACKEND
# ====================================================================

def run_all_experiments(backend):
    """Run all 6 experiments on a single backend."""
    bname = backend.name
    bresults = {"backend": bname, "num_qubits": backend.num_qubits}

    # ---- EXP 1: COHERENT OVER-ROTATION ----
    print("\n  [EXP 1] Coherent Over-Rotation (predicted alpha ~ 2)")
    e1 = {}
    for d in COHERENT_DEPTHS:
        qc = build_coherent_drift(N_QUBITS_DEPTH, d, COHERENT_EPS)
        target = "0" * N_QUBITS_DEPTH
        print("    depth=" + str(d) + " ", end="", flush=True)
        try:
            c, jid = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            e1[d] = {"fidelity": f, "infidelity": 1 - f, "job_id": jid}
            print("F=" + str(round(f, 4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            e1[d] = {"error": str(e)[:200]}
    bresults["exp1_coherent"] = {str(k): v for k, v in e1.items()}

    # ---- EXP 2: RANDOM CLIFFORD DEPTH ----
    print("\n  [EXP 2] Clifford Depth Scaling (opt_level=0)")
    e2 = {}
    for d in DEPTHS:
        qc = build_clifford_depth(N_QUBITS_DEPTH, d)
        target = "0" * N_QUBITS_DEPTH
        print("    depth=" + str(d) + " ", end="", flush=True)
        try:
            c, jid = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            e2[d] = {"fidelity": f, "infidelity": 1 - f, "job_id": jid}
            print("F=" + str(round(f, 4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            e2[d] = {"error": str(e)[:200]}
    bresults["exp2_clifford"] = {str(k): v for k, v in e2.items()}

    # ---- EXP 3: GHZ SCALING ----
    print("\n  [EXP 3] GHZ Qubit Scaling")
    e3 = {}
    for nq in GHZ_QUBITS:
        if nq > backend.num_qubits:
            continue
        qc = build_ghz(nq)
        target = ["0" * nq, "1" * nq]
        print("    GHZ-" + str(nq) + "Q ", end="", flush=True)
        try:
            c, jid = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            e3[nq] = {"fidelity": f, "infidelity": 1 - f, "job_id": jid}
            print("F=" + str(round(f, 4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            e3[nq] = {"error": str(e)[:200]}
    bresults["exp3_ghz"] = {str(k): v for k, v in e3.items()}

    # ---- EXP 4: NOISE AMPLIFICATION ----
    print("\n  [EXP 4] Noise Amplification (repeated unit circuits)")
    e4 = {}
    for r in NOISE_AMP_REPEATS:
        qc = build_repeated_unit(N_QUBITS_DEPTH, base_depth=5, repeats=r)
        target = "0" * N_QUBITS_DEPTH
        eff_depth = 5 * 2 * r
        print("    repeats=" + str(r) + " (eff_d=" + str(eff_depth) + ") ", end="", flush=True)
        try:
            c, jid = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            e4[r] = {"fidelity": f, "infidelity": 1 - f,
                     "effective_depth": eff_depth, "job_id": jid}
            print("F=" + str(round(f, 4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            e4[r] = {"error": str(e)[:200]}
    bresults["exp4_amplification"] = {str(k): v for k, v in e4.items()}

    # ---- EXP 5: ECHO (X-X identity) ----
    print("\n  [EXP 5] Echo Experiment (X-X identity, gate quality)")
    e5 = {}
    for d in ECHO_DEPTHS:
        qc = build_echo_circuit(N_QUBITS_DEPTH, d)
        target = "0" * N_QUBITS_DEPTH
        print("    depth=" + str(d) + " ", end="", flush=True)
        try:
            c, jid = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            e5[d] = {"fidelity": f, "infidelity": 1 - f, "job_id": jid}
            print("F=" + str(round(f, 4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            e5[d] = {"error": str(e)[:200]}
    bresults["exp5_echo"] = {str(k): v for k, v in e5.items()}

    # ---- EXP 6: LONG-DEPTH RANDOM (saturation) ----
    print("\n  [EXP 6] Long-Depth Random Circuits (saturation analysis)")
    e6 = {}
    for d in LONG_DEPTHS:
        qc = build_random_long(N_QUBITS_DEPTH, d)
        target = "0" * N_QUBITS_DEPTH
        print("    depth=" + str(d) + " ", end="", flush=True)
        try:
            c, jid = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            e6[d] = {"fidelity": f, "infidelity": 1 - f, "job_id": jid}
            print("F=" + str(round(f, 4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            e6[d] = {"error": str(e)[:200]}
    bresults["exp6_long"] = {str(k): v for k, v in e6.items()}

    return bresults

# ====================================================================
# K-R ANALYSIS PER BACKEND
# ====================================================================

def analyze_backend(br):
    """Fit K-R to each experiment and add to results."""
    bname = br["backend"]
    print("\n  K-R ANALYSIS: " + bname)
    print("  " + "-" * 60)

    fits = {}

    # EXP 1 fit
    e1 = br.get("exp1_coherent", {})
    valid = [(int(k), v["infidelity"]) for k, v in e1.items() if "infidelity" in v]
    if len(valid) >= 4:
        ds, cs = zip(*sorted(valid))
        K = 1.0 / np.array(ds)
        f = fit_kr(K, np.array(cs))
        fits["exp1_coherent_kr"] = f
        print("  EXP 1 (coherent):       alpha=" + str(round(f.get("alpha", 0), 4)) +
              ", R2=" + str(round(f.get("R2", 0), 4)))

    # EXP 2 fit
    e2 = br.get("exp2_clifford", {})
    valid = [(int(k), v["infidelity"]) for k, v in e2.items() if "infidelity" in v]
    if len(valid) >= 4:
        ds, cs = zip(*sorted(valid))
        K = 1.0 / np.array(ds)
        f = fit_kr(K, np.array(cs))
        fits["exp2_clifford_kr"] = f
        print("  EXP 2 (Clifford depth): alpha=" + str(round(f.get("alpha", 0), 4)) +
              ", R2=" + str(round(f.get("R2", 0), 4)))

    # EXP 3 fit (qubit scaling: K = 1/n_qubits)
    e3 = br.get("exp3_ghz", {})
    valid = [(int(k), v["infidelity"]) for k, v in e3.items() if "infidelity" in v]
    if len(valid) >= 4:
        ns, cs = zip(*sorted(valid))
        K = 1.0 / np.array(ns)
        f = fit_kr(K, np.array(cs))
        fits["exp3_ghz_kr"] = f
        print("  EXP 3 (GHZ qubits):     alpha=" + str(round(f.get("alpha", 0), 4)) +
              ", R2=" + str(round(f.get("R2", 0), 4)))

    # EXP 4 fit (effective depth)
    e4 = br.get("exp4_amplification", {})
    valid = [(v.get("effective_depth", 0), v["infidelity"])
             for v in e4.values() if "infidelity" in v]
    if len(valid) >= 3:
        ds, cs = zip(*sorted(valid))
        K = 1.0 / np.array(ds)
        f = fit_kr(K, np.array(cs))
        fits["exp4_amp_kr"] = f
        print("  EXP 4 (amplification):  alpha=" + str(round(f.get("alpha", 0), 4)) +
              ", R2=" + str(round(f.get("R2", 0), 4)))

    # EXP 5 fit (echo)
    e5 = br.get("exp5_echo", {})
    valid = [(int(k), v["infidelity"]) for k, v in e5.items() if "infidelity" in v]
    if len(valid) >= 4:
        ds, cs = zip(*sorted(valid))
        K = 1.0 / np.array(ds)
        f = fit_kr(K, np.array(cs))
        fits["exp5_echo_kr"] = f
        print("  EXP 5 (echo):           alpha=" + str(round(f.get("alpha", 0), 4)) +
              ", R2=" + str(round(f.get("R2", 0), 4)))

    # EXP 6 fit (long depth)
    e6 = br.get("exp6_long", {})
    valid = [(int(k), v["infidelity"]) for k, v in e6.items() if "infidelity" in v]
    if len(valid) >= 3:
        ds, cs = zip(*sorted(valid))
        K = 1.0 / np.array(ds)
        f = fit_kr(K, np.array(cs))
        fits["exp6_long_kr"] = f
        print("  EXP 6 (long depth):     alpha=" + str(round(f.get("alpha", 0), 4)) +
              ", R2=" + str(round(f.get("R2", 0), 4)))

    br["kr_fits"] = fits
    return br

# ====================================================================
# RUN ALL
# ====================================================================

all_results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "shots": SHOTS,
    "n_qubits_depth": N_QUBITS_DEPTH,
    "coherent_eps": COHERENT_EPS,
    "backends": {}
}

for backend in backends_to_use:
    print("\n" + "#" * 72)
    print("# RUNNING ALL 6 EXPERIMENTS ON: " + backend.name)
    print("#" * 72)

    br = run_all_experiments(backend)
    br = analyze_backend(br)
    all_results["backends"][backend.name] = br

# ====================================================================
# CROSS-BACKEND COMPARISON
# ====================================================================

print("\n" + "=" * 72)
print("  CROSS-BACKEND ALPHA COMPARISON")
print("=" * 72)

exp_names = ["exp1_coherent_kr", "exp2_clifford_kr", "exp3_ghz_kr",
             "exp4_amp_kr", "exp5_echo_kr", "exp6_long_kr"]
exp_labels = ["Coherent", "Clifford", "GHZ", "NoiseAmp", "Echo", "LongRand"]

print("\n  " + "Backend".ljust(20) + " | " +
      " | ".join(l.ljust(8) for l in exp_labels))
print("  " + "-" * 90)

for bname, br in all_results["backends"].items():
    fits = br.get("kr_fits", {})
    row_vals = []
    for ek in exp_names:
        a = fits.get(ek, {}).get("alpha")
        if a is not None:
            row_vals.append(str(round(a, 3)).ljust(8))
        else:
            row_vals.append("--".ljust(8))
    print("  " + bname.ljust(20) + " | " + " | ".join(row_vals))

# ====================================================================
# SAVE
# ====================================================================

def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean(x) for x in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    return obj

with open(SAVE_FILE, "w") as f:
    json.dump(clean(all_results), f, indent=2)
print("\n  Saved: " + SAVE_FILE)

# ====================================================================
# FIGURES
# ====================================================================

print("\n  Generating figures...")
primary = backends_to_use[0]
br_p = all_results["backends"][primary.name]

# FIGURE 1: 6-panel results from primary backend
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("K-R 6-Experiment Validation on Real IBM Quantum: " + primary.name,
             fontsize=14, fontweight="bold")

panels = [
    ("exp1_coherent", "exp1_coherent_kr", COHERENT_DEPTHS,
     "(a) Coherent Over-Rotation", "Depth", "Infidelity"),
    ("exp2_clifford", "exp2_clifford_kr", DEPTHS,
     "(b) Clifford Depth Scaling", "Depth", "Infidelity"),
    ("exp3_ghz", "exp3_ghz_kr", GHZ_QUBITS,
     "(c) GHZ Qubit Scaling", "Number of Qubits", "GHZ Infidelity"),
    ("exp4_amplification", "exp4_amp_kr", None,
     "(d) Noise Amplification", "Effective Depth", "Infidelity"),
    ("exp5_echo", "exp5_echo_kr", ECHO_DEPTHS,
     "(e) Echo (X-X identity)", "Depth", "Infidelity"),
    ("exp6_long", "exp6_long_kr", LONG_DEPTHS,
     "(f) Long-Depth Saturation", "Depth", "Infidelity"),
]

for ax, (ek, fk, _, title, xl, yl) in zip(axes.flat, panels):
    data = br_p.get(ek, {})
    fit = br_p.get("kr_fits", {}).get(fk, {})

    if ek == "exp4_amplification":
        valid = [(v.get("effective_depth", 0), v["infidelity"])
                 for v in data.values() if "infidelity" in v]
    else:
        valid = [(int(k), v["infidelity"]) for k, v in data.items() if "infidelity" in v]

    if valid:
        xs, ys = zip(*sorted(valid))
        ax.plot(xs, ys, "ro-", lw=2, markersize=8, label="Real hardware")

        if fit.get("alpha") and fit.get("R2", 0) > 0:
            xs_smooth = np.linspace(min(xs), max(xs), 100)
            K_smooth = 1.0 / xs_smooth
            cf = fit["C0"] / (K_smooth ** fit["alpha"]) + fit["R"]
            ax.plot(xs_smooth, cf, "k--", lw=2,
                    label="K-R: alpha=" + str(round(fit["alpha"], 3)) +
                          ", R^2=" + str(round(fit["R2"], 3)))

    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("Fig_IBM_8_Regimes_" + primary.name + ".png", dpi=300, bbox_inches="tight")
print("  Saved: Fig_IBM_8_Regimes_" + primary.name + ".png")

# FIGURE 2: Cross-backend alpha comparison
fig2, ax = plt.subplots(1, 1, figsize=(13, 7))
n_exps = len(exp_labels)
n_backends = len(all_results["backends"])
bar_w = 0.8 / n_backends
x = np.arange(n_exps)

for i, (bname, br) in enumerate(all_results["backends"].items()):
    fits = br.get("kr_fits", {})
    alphas = [fits.get(ek, {}).get("alpha", 0) or 0 for ek in exp_names]
    ax.bar(x + i*bar_w, alphas, bar_w, label=bname, alpha=0.85, edgecolor="black")

ax.set_xticks(x + bar_w * (n_backends-1) / 2)
ax.set_xticklabels(exp_labels)
ax.set_ylabel("K-R Noise Exponent (alpha)", fontsize=12)
ax.set_title("alpha as Multi-Regime Noise Classifier on Real IBM Hardware",
             fontsize=13, fontweight="bold")
ax.axhline(1.0, color="black", ls=":", lw=1, alpha=0.5, label="alpha=1 (linear)")
ax.axhline(2.0, color="red", ls=":", lw=1, alpha=0.5, label="alpha=2 (coherent)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
fig2.tight_layout()
fig2.savefig("Fig_IBM_8_Regimes_CrossBackend.png", dpi=300, bbox_inches="tight")
print("  Saved: Fig_IBM_8_Regimes_CrossBackend.png")

# FIGURE 3: Direct comparison plots (all backends overlaid)
if len(all_results["backends"]) > 1:
    fig3, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig3.suptitle("Multi-Backend Comparison Across 6 Experiments",
                  fontsize=14, fontweight="bold")

    for ax, (ek, fk, _, title, xl, yl) in zip(axes.flat, panels):
        for bname, br in all_results["backends"].items():
            data = br.get(ek, {})
            if ek == "exp4_amplification":
                valid = [(v.get("effective_depth", 0), v["infidelity"])
                         for v in data.values() if "infidelity" in v]
            else:
                valid = [(int(k), v["infidelity"]) for k, v in data.items() if "infidelity" in v]
            if valid:
                xs, ys = zip(*sorted(valid))
                ax.plot(xs, ys, "o-", lw=2, markersize=7, label=bname)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    fig3.savefig("Fig_IBM_8_Regimes_MultiBackend.png", dpi=300, bbox_inches="tight")
    print("  Saved: Fig_IBM_8_Regimes_MultiBackend.png")

elapsed = time.time() - start_time
print("\n" + "=" * 72)
print("  COMPLETE! Total time: " + str(round(elapsed/60, 1)) + " minutes")
print("\n  Upload to Claude:")
print("    1. ibm_8_regimes_results.json")
print("    2. Fig_IBM_8_Regimes_" + primary.name + ".png")
print("    3. Fig_IBM_8_Regimes_CrossBackend.png")
if len(all_results["backends"]) > 1:
    print("    4. Fig_IBM_8_Regimes_MultiBackend.png")
print("=" * 72)