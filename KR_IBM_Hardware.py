##############################################################################
#  KR_IBM_Hardware.py - FIXED for new IBM Quantum Platform (2025+)
#
#  STEPS:
#  1. Put your API token between the quotes on line 18 below
#  2. Save this file
#  3. Run: python KR_IBM_Hardware.py
##############################################################################

MY_TOKEN = "    "

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
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.primitives import StatevectorSampler

SHOTS = 4096
SAVE_FILE = "ibm_hardware_results.json"
start_time = time.time()

print("=" * 60)
print("  K-R BENCHMARKING ON REAL IBM QUANTUM HARDWARE")
print("=" * 60)

if MY_TOKEN == "PUT_TOKEN_HERE":
    print("\n  ERROR: Paste your token on line 18, save, run again.\n")
    exit()

# ---- CONNECT (using NEW IBM Quantum Platform channel) ----
print("\n[1/6] Connecting to IBM Quantum Platform...")
service = None

# Try new channel first (ibm_quantum_platform), then fallback to ibm_cloud
for channel_name in ["ibm_quantum_platform", "ibm_cloud"]:
    try:
        print("  Trying channel: " + channel_name)
        QiskitRuntimeService.save_account(
            channel=channel_name,
            token=MY_TOKEN,
            overwrite=True,
            set_as_default=True
        )
        service = QiskitRuntimeService(channel=channel_name)
        print("  Connected via " + channel_name + "!")
        break
    except Exception as e:
        print("  " + channel_name + " failed: " + str(e)[:100])
        continue

if service is None:
    print("\n  All connection attempts failed.")
    print("  Check that your API key is correct.")
    exit()

# ---- SELECT BACKEND ----
print("\n[2/6] Finding available quantum computer...")
try:
    backend = service.least_busy(min_num_qubits=8, operational=True, simulator=False)
    print("  Selected: " + backend.name + " (" + str(backend.num_qubits) + " qubits)")
except Exception as e:
    print("  Auto-select with simulator filter failed: " + str(e)[:100])
    try:
        backend = service.least_busy(min_num_qubits=8, operational=True)
        print("  Selected: " + backend.name + " (" + str(backend.num_qubits) + " qubits)")
    except Exception as e2:
        print("  Listing all backends...")
        backends = service.backends()
        if not backends:
            print("  No backends available!")
            exit()
        # Pick first non-simulator if possible
        backend = None
        for b in backends:
            try:
                if b.num_qubits >= 8 and "simulator" not in b.name.lower():
                    backend = b
                    break
            except:
                continue
        if backend is None:
            backend = backends[0]
        print("  Using: " + backend.name)

backend_name = backend.name

# ---- BUILD GHZ CIRCUITS ----
print("\n[3/6] Building GHZ circuits (2-8 qubits)...")
ghz_circuits = {}
for nq in [2, 3, 4, 5, 6, 7, 8]:
    qc = QuantumCircuit(nq)
    qc.h(0)
    for i in range(nq - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    ghz_circuits[nq] = qc
    print("  Built " + str(nq) + "-qubit GHZ")

# ---- BUILD DEPTH CIRCUITS ----
print("\n[4/6] Building depth-scaling circuits (5 qubits)...")
depth_circuits = {}
for depth in [1, 2, 3, 5, 8, 10, 15, 20]:
    qc = QuantumCircuit(5)
    for d in range(depth):
        for q in range(5):
            qc.h(q)
        for q in range(0, 4, 2):
            qc.cx(q, q + 1)
        if d % 2 == 1:
            for q in range(1, 4, 2):
                qc.cx(q, q + 1)
    qc_inv = qc.inverse()
    qc_full = qc.compose(qc_inv)
    qc_full.measure_all()
    depth_circuits[depth] = qc_full
    print("  Built depth-" + str(depth) + " mirror circuit")

# ---- RUN ON REAL HARDWARE ----
print("\n[5/6] RUNNING ON REAL IBM QUANTUM HARDWARE")
print("  Each circuit takes 1-15 minutes (queue time). Please wait...\n")

sampler = SamplerV2(mode=backend)

ghz_results = {}
for nq in sorted(ghz_circuits.keys()):
    qc = ghz_circuits[nq]
    print("  Submitting " + str(nq) + "-qubit GHZ...", end=" ", flush=True)
    try:
        qc_t = transpile(qc, backend=backend, optimization_level=1)
        job = sampler.run([qc_t], shots=SHOTS)
        print("Job ID: " + job.job_id())
        print("    Waiting for result...", end=" ", flush=True)
        result = job.result()
        # Get counts - handle different result formats
        try:
            counts = result[0].data.meas.get_counts()
        except:
            counts = result[0].data[list(result[0].data.keys())[0]].get_counts()
        all_zero = "0" * nq
        all_one = "1" * nq
        fid = (counts.get(all_zero, 0) + counts.get(all_one, 0)) / SHOTS
        ghz_results[nq] = {"fidelity": fid, "infidelity": 1 - fid}
        print("F=" + str(round(fid, 4)) + ", Infid=" + str(round(1 - fid, 4)))
    except Exception as e:
        print("Error: " + str(e)[:150])
        ghz_results[nq] = {"fidelity": 0, "infidelity": 1, "error": str(e)}

depth_results = {}
for depth in sorted(depth_circuits.keys()):
    qc = depth_circuits[depth]
    print("  Submitting depth-" + str(depth) + " circuit...", end=" ", flush=True)
    try:
        qc_t = transpile(qc, backend=backend, optimization_level=1)
        job = sampler.run([qc_t], shots=SHOTS)
        print("Job ID: " + job.job_id())
        print("    Waiting for result...", end=" ", flush=True)
        result = job.result()
        try:
            counts = result[0].data.meas.get_counts()
        except:
            counts = result[0].data[list(result[0].data.keys())[0]].get_counts()
        fid = counts.get("00000", 0) / SHOTS
        depth_results[depth] = {"fidelity": fid, "infidelity": 1 - fid}
        print("F=" + str(round(fid, 4)) + ", Infid=" + str(round(1 - fid, 4)))
    except Exception as e:
        print("Error: " + str(e)[:150])
        depth_results[depth] = {"fidelity": 0, "infidelity": 1, "error": str(e)}

# ---- LOCAL SIMULATOR ----
print("\n  Running ideal simulator for comparison...")
sim_sampler = StatevectorSampler()

ghz_sim = {}
for nq in sorted(ghz_circuits.keys()):
    r = sim_sampler.run([ghz_circuits[nq]], shots=SHOTS).result()
    c = r[0].data.meas.get_counts()
    f = (c.get("0" * nq, 0) + c.get("1" * nq, 0)) / SHOTS
    ghz_sim[nq] = {"fidelity": f, "infidelity": 1 - f}

depth_sim = {}
for depth in sorted(depth_circuits.keys()):
    r = sim_sampler.run([depth_circuits[depth]], shots=SHOTS).result()
    c = r[0].data.meas.get_counts()
    f = c.get("00000", 0) / SHOTS
    depth_sim[depth] = {"fidelity": f, "infidelity": 1 - f}

# ---- K-R ANALYSIS ----
print("\n[6/6] K-R ANALYSIS ON REAL HARDWARE DATA")
print("=" * 60)

print("\n  GHZ Circuit Results:")
print("  Qubits  Hardware  Simulator")
for nq in sorted(ghz_results.keys()):
    print("  " + str(nq) + "       " + str(round(ghz_results[nq]["infidelity"], 4)) +
          "    " + str(round(ghz_sim[nq]["infidelity"], 4)))

print("\n  Depth Scaling Results:")
print("  Depth   Hardware  Simulator")
for d in sorted(depth_results.keys()):
    print("  " + str(d) + "       " + str(round(depth_results[d]["infidelity"], 4)) +
          "    " + str(round(depth_sim[d]["infidelity"], 4)))

def kr_model(K, C0, R, alpha):
    return C0 / (np.clip(K, 1e-30, None) ** alpha) + R

depths_arr = np.array(sorted(depth_results.keys()))
C_hw = np.array([depth_results[d]["infidelity"] for d in sorted(depth_results.keys())])
K_arr = 1.0 / depths_arr

kr_alpha = None
kr_r2 = None
popt = None
try:
    popt, _ = curve_fit(kr_model, K_arr, C_hw,
                        p0=[0.5, 0.01, 1.0],
                        bounds=([0, 0, 0.01], [50, 2, 15]),
                        maxfev=50000)
    C0_fit, R_fit, alpha_fit = popt
    y_pred = kr_model(K_arr, *popt)
    ss_r = np.sum((C_hw - y_pred) ** 2)
    ss_t = np.sum((C_hw - C_hw.mean()) ** 2)
    r2 = 1 - ss_r / ss_t if ss_t > 1e-30 else 0
    kr_alpha = float(alpha_fit)
    kr_r2 = float(r2)
    print("\n  ===================================")
    print("  K-R FIT TO REAL IBM QUANTUM DATA:")
    print("    alpha = " + str(round(alpha_fit, 4)))
    print("    C0    = " + str(round(C0_fit, 4)))
    print("    R     = " + str(round(R_fit, 4)))
    print("    R^2   = " + str(round(r2, 4)))
    print("  ===================================")
except Exception as e:
    print("  K-R fit error: " + str(e))

# ---- SAVE ----
output = {
    "backend": backend_name,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "shots": SHOTS,
    "ghz_hardware": {str(k): {"fidelity": v["fidelity"], "infidelity": v["infidelity"]}
                     for k, v in ghz_results.items()},
    "ghz_simulator": {str(k): {"fidelity": v["fidelity"], "infidelity": v["infidelity"]}
                      for k, v in ghz_sim.items()},
    "depth_hardware": {str(k): {"fidelity": v["fidelity"], "infidelity": v["infidelity"]}
                       for k, v in depth_results.items()},
    "depth_simulator": {str(k): {"fidelity": v["fidelity"], "infidelity": v["infidelity"]}
                        for k, v in depth_sim.items()},
    "kr_fit": {"alpha": kr_alpha, "R2": kr_r2},
}
with open(SAVE_FILE, "w") as f:
    json.dump(output, f, indent=2)
print("\n  Saved: " + SAVE_FILE)

# ---- FIGURE ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("K-R Benchmarking on Real IBM Quantum: " + backend_name,
             fontsize=13, fontweight="bold")

ax = axes[0]
nqs = sorted(ghz_results.keys())
ax.plot(nqs, [ghz_results[n]["infidelity"] for n in nqs],
        "ro-", lw=2, markersize=8, label="Real (" + backend_name + ")")
ax.plot(nqs, [ghz_sim[n]["infidelity"] for n in nqs],
        "bs--", lw=2, markersize=6, label="Ideal simulator")
ax.set_xlabel("Number of Qubits")
ax.set_ylabel("GHZ Infidelity")
ax.set_title("(a) GHZ Infidelity: Real vs Ideal")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(depths_arr, C_hw, "ro-", lw=2, markersize=8, label="Real hardware")
ax.plot(depths_arr, [depth_sim[d]["infidelity"] for d in sorted(depth_sim.keys())],
        "bs--", lw=2, markersize=6, label="Ideal simulator")
if popt is not None:
    d_smooth = np.linspace(1, max(depths_arr), 100)
    K_smooth = 1.0 / d_smooth
    ax.plot(d_smooth, kr_model(K_smooth, *popt), "k--", lw=2,
            label="K-R fit (alpha=" + str(round(kr_alpha, 3)) + ")")
ax.set_xlabel("Circuit Depth")
ax.set_ylabel("Mirror Circuit Infidelity")
ax.set_title("(b) Depth Scaling + K-R Fit (5Q)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("Fig_IBM_Hardware_Results.png", dpi=300, bbox_inches="tight")
print("  Saved: Fig_IBM_Hardware_Results.png")

elapsed = time.time() - start_time
print("\n" + "=" * 60)
print("  DONE! Total time: " + str(round(elapsed / 60, 1)) + " minutes")
print("\n  Upload these files to Claude:")
print("    1. ibm_hardware_results.json")
print("    2. Fig_IBM_Hardware_Results.png")
print("=" * 60)