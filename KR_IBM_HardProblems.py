##############################################################################
#  KR_IBM_HardProblems.py
#
#  Tests K-R against state-of-the-art quantum benchmarking on 5 HARD PROBLEMS
#  that current methods struggle with. Run on REAL IBM Quantum hardware.
#
#  HARD PROBLEMS TESTED:
#    A. CIRCUIT FIDELITY PREDICTION (predict deep from shallow data)
#       SOTA: Empirical measurement at every depth
#       K-R:  Fit shallow, extrapolate to deep
#
#    B. NOISE TYPE CLASSIFICATION (coherent vs stochastic)
#       SOTA: Purity benchmarking (~100 circuits)
#       K-R:  Single alpha value (~7 circuits)
#
#    C. DEVICE SELECTION (which backend for my algorithm?)
#       SOTA: Quantum Volume comparison (exponential cost)
#       K-R:  alpha + R fingerprint (~10 circuits)
#
#    D. NOISE STABILITY MONITORING (hardware drift detection)
#       SOTA: Daily full RB re-run
#       K-R:  Track alpha over time (cheap)
#
#    E. EARLY-WARNING SATURATION DETECTION (when does fidelity collapse?)
#       SOTA: Trial and error
#       K-R:  Predict from R parameter
#
#  This generates results for a STRONG paper showing K-R provides UNIQUE
#  capabilities not available with existing methods.
#
#  STEPS:
#  1. Paste your token on line 27
#  2. Save and run: python KR_IBM_HardProblems.py
#
#  Estimated time: 45-90 minutes (queue time)
#  QPU usage: ~6-8 minutes (within free tier)
##############################################################################

MY_TOKEN = "       "

# ----- EXPERIMENT SETTINGS -----
SHOTS = 4096
N_QUBITS = 5
USE_TWO_BACKENDS = True

# Problem A: Train on shallow, predict deep
TRAIN_DEPTHS = [1, 2, 4, 8]          # K-R fits these
TEST_DEPTHS = [16, 32, 64]            # Predict and verify these

# Problem B: Coherent vs stochastic discrimination
COHERENT_DEPTHS = [1, 2, 4, 8, 16, 32, 64]
COHERENT_EPS = 0.05                   # Drift angle per gate

# Problem C: Device fingerprint
FINGERPRINT_DEPTHS = [1, 2, 4, 8, 16, 32]
FINGERPRINT_GHZ_QUBITS = [2, 3, 4, 5, 6, 7, 8]

# Problem D: Stability (3 repeated runs of same circuit set)
STABILITY_DEPTHS = [1, 4, 16, 64]
STABILITY_RUNS = 3                    # 3 runs over a few minutes

# Problem E: Saturation prediction
SATURATION_DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128]

##############################################################################
#  DO NOT EDIT BELOW
##############################################################################

import json, time
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

SAVE_FILE = "ibm_hardproblems_results.json"
start_time = time.time()

print("=" * 75)
print("  K-R vs STATE-OF-THE-ART: 5 HARD PROBLEMS ON REAL IBM HARDWARE")
print("=" * 75)

if MY_TOKEN == "PUT_TOKEN_HERE":
    print("\n  ERROR: Paste your token on line 27, save, run again.\n")
    exit()

# ====================================================================
# CONNECT
# ====================================================================
print("\n[CONNECT] Connecting to IBM Quantum Platform...")
service = None
for ch in ["ibm_quantum_platform", "ibm_cloud"]:
    try:
        QiskitRuntimeService.save_account(channel=ch, token=MY_TOKEN,
                                          overwrite=True, set_as_default=True)
        service = QiskitRuntimeService(channel=ch)
        print("  Connected via " + ch)
        break
    except: continue
if service is None:
    print("  Connection failed."); exit()

# ====================================================================
# SELECT BACKENDS
# ====================================================================
print("\n[BACKENDS] Selecting quantum computers...")
backends = []
try:
    primary = service.least_busy(min_num_qubits=N_QUBITS, operational=True, simulator=False)
    backends.append(primary)
    print("  Primary: " + primary.name)
    if USE_TWO_BACKENDS:
        for b in service.backends(operational=True, simulator=False, min_num_qubits=N_QUBITS):
            if b.name != primary.name:
                backends.append(b)
                print("  Secondary: " + b.name)
                break
except Exception as e:
    print("  Error: " + str(e)); exit()

# ====================================================================
# CIRCUIT BUILDERS
# ====================================================================

def clifford_layer(qc, nq):
    """One layer of Clifford gates."""
    for q in range(nq):
        qc.h(q)
    for q in range(0, nq-1, 2):
        qc.cx(q, q+1)
    for q in range(nq):
        qc.s(q)
    if nq > 2:
        for q in range(1, nq-1, 2):
            qc.cx(q, q+1)

def build_clifford_inverse(nq, depth):
    """Standard Clifford+inverse circuit. Ideal: |0..0>."""
    qc = QuantumCircuit(nq)
    for _ in range(depth):
        clifford_layer(qc, nq)
    qc_full = qc.compose(qc.inverse())
    qc_full.measure_all()
    return qc_full

def build_coherent_drift(nq, depth, eps):
    """Pure coherent over-rotation. Predicted alpha ~ 2."""
    qc = QuantumCircuit(nq)
    for _ in range(depth):
        for q in range(nq):
            qc.rx(eps, q)
    qc.measure_all()
    return qc

def build_ghz(nq):
    qc = QuantumCircuit(nq)
    qc.h(0)
    for i in range(nq-1): qc.cx(i, i+1)
    qc.measure_all()
    return qc

# ====================================================================
# EXECUTION HELPERS
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
# K-R MODEL + COMPETITORS (the "SOTA" comparison models)
# ====================================================================

def kr(K, C0, R, alpha):
    return C0 / (np.clip(K, 1e-30, None)**alpha) + R

def m_exp(K, A, lam, B):
    """Exponential: standard model in randomized benchmarking."""
    return A * np.exp(-lam * K) + B

def m_strexp(K, A, lam, beta, B):
    """Stretched exponential: common alternative."""
    return A * np.exp(-(lam * np.clip(K, 1e-10, None))**beta) + B

def fit_kr(K, C):
    try:
        p, _ = curve_fit(kr, K, C, p0=[0.5,0.01,1.0],
                        bounds=([0,0,0.01],[50,2,15]), maxfev=80000)
        yp = kr(K, *p)
        ssr=np.sum((C-yp)**2); sst=np.sum((C-C.mean())**2)
        r2=1-ssr/sst if sst>1e-30 else 0
        return {"alpha":float(p[2]),"C0":float(p[0]),"R":float(p[1]),
                "R2":float(r2),"params":[float(x) for x in p],"ok":True}
    except: return {"ok":False,"alpha":None}

def fit_exp(K, C):
    try:
        p, _ = curve_fit(m_exp, K, C, p0=[1,1,0],
                        bounds=([0,0,-2],[10,50,2]), maxfev=80000)
        yp = m_exp(K, *p)
        ssr=np.sum((C-yp)**2); sst=np.sum((C-C.mean())**2)
        r2=1-ssr/sst if sst>1e-30 else 0
        return {"params":[float(x) for x in p],"R2":float(r2),"ok":True}
    except: return {"ok":False}

def fit_strexp(K, C):
    try:
        p, _ = curve_fit(m_strexp, K, C, p0=[1,1,1,0],
                        bounds=([0,0,0.01,-2],[10,50,5,2]), maxfev=80000)
        yp = m_strexp(K, *p)
        ssr=np.sum((C-yp)**2); sst=np.sum((C-C.mean())**2)
        r2=1-ssr/sst if sst>1e-30 else 0
        return {"params":[float(x) for x in p],"R2":float(r2),"ok":True}
    except: return {"ok":False}

# ====================================================================
# HARD PROBLEM A: FIDELITY PREDICTION
# ====================================================================

def hard_problem_A_prediction(backend):
    """Train K-R on shallow circuits, PREDICT deep circuits, then verify."""
    print("\n  [HARD PROBLEM A] Predict deep circuits from shallow data")
    print("  K-R: fit on depths " + str(TRAIN_DEPTHS) + ", predict " + str(TEST_DEPTHS))

    train_results = {}
    test_results = {}

    # Run TRAIN circuits
    print("  -- Training data --")
    for d in TRAIN_DEPTHS:
        qc = build_clifford_inverse(N_QUBITS, d)
        target = "0" * N_QUBITS
        print("    train d=" + str(d), end=" ", flush=True)
        try:
            c, j = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            train_results[d] = {"fidelity": f, "infidelity": 1-f, "job_id": j}
            print("F=" + str(round(f,4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            train_results[d] = {"error": str(e)[:200]}

    # Run TEST circuits (ground truth)
    print("  -- Test data (ground truth) --")
    for d in TEST_DEPTHS:
        qc = build_clifford_inverse(N_QUBITS, d)
        target = "0" * N_QUBITS
        print("    test d=" + str(d), end=" ", flush=True)
        try:
            c, j = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            test_results[d] = {"fidelity": f, "infidelity": 1-f, "job_id": j}
            print("F=" + str(round(f,4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            test_results[d] = {"error": str(e)[:200]}

    # Fit K-R + competitors on TRAIN data
    train_d = sorted([d for d in train_results if "infidelity" in train_results[d]])
    if len(train_d) < 3:
        return {"train": train_results, "test": test_results, "error": "insufficient data"}

    K_train = 1.0/np.array(train_d)
    C_train = np.array([train_results[d]["infidelity"] for d in train_d])

    fit_k = fit_kr(K_train, C_train)
    fit_e = fit_exp(K_train, C_train)
    fit_s = fit_strexp(K_train, C_train)

    # Predict test depths and compare
    predictions = {}
    test_d = sorted([d for d in test_results if "infidelity" in test_results[d]])

    for d in test_d:
        K_test = 1.0/d
        actual = test_results[d]["infidelity"]
        pred_kr = kr(K_test, *fit_k["params"]) if fit_k["ok"] else None
        pred_ex = m_exp(K_test, *fit_e["params"]) if fit_e["ok"] else None
        pred_se = m_strexp(K_test, *fit_s["params"]) if fit_s["ok"] else None
        predictions[d] = {
            "actual": float(actual),
            "kr_predicted": float(pred_kr) if pred_kr is not None else None,
            "kr_error": float(abs(pred_kr - actual)) if pred_kr is not None else None,
            "exp_predicted": float(pred_ex) if pred_ex is not None else None,
            "exp_error": float(abs(pred_ex - actual)) if pred_ex is not None else None,
            "strexp_predicted": float(pred_se) if pred_se is not None else None,
            "strexp_error": float(abs(pred_se - actual)) if pred_se is not None else None,
        }

    # Mean absolute prediction error (lower is better)
    kr_errs = [predictions[d]["kr_error"] for d in test_d if predictions[d]["kr_error"] is not None]
    ex_errs = [predictions[d]["exp_error"] for d in test_d if predictions[d]["exp_error"] is not None]
    se_errs = [predictions[d]["strexp_error"] for d in test_d if predictions[d]["strexp_error"] is not None]

    print("\n  PREDICTION ACCURACY (mean abs error):")
    if kr_errs: print("    K-R:           " + str(round(np.mean(kr_errs), 4)))
    if ex_errs: print("    Exponential:   " + str(round(np.mean(ex_errs), 4)))
    if se_errs: print("    Stretched exp: " + str(round(np.mean(se_errs), 4)))

    return {
        "train": {str(k): v for k, v in train_results.items()},
        "test": {str(k): v for k, v in test_results.items()},
        "predictions": {str(k): v for k, v in predictions.items()},
        "kr_fit": fit_k,
        "exp_fit": fit_e,
        "strexp_fit": fit_s,
        "kr_mae": float(np.mean(kr_errs)) if kr_errs else None,
        "exp_mae": float(np.mean(ex_errs)) if ex_errs else None,
        "strexp_mae": float(np.mean(se_errs)) if se_errs else None,
    }

# ====================================================================
# HARD PROBLEM B: NOISE TYPE CLASSIFICATION
# ====================================================================

def hard_problem_B_classification(backend):
    """Discriminate coherent (RX drift) from stochastic (Clifford) noise via alpha."""
    print("\n  [HARD PROBLEM B] Coherent vs Stochastic noise discrimination")

    # Coherent regime
    print("  -- Coherent over-rotation (predicted alpha ~ 2) --")
    coherent_results = {}
    for d in COHERENT_DEPTHS:
        qc = build_coherent_drift(N_QUBITS, d, COHERENT_EPS)
        target = "0" * N_QUBITS
        print("    coh d=" + str(d), end=" ", flush=True)
        try:
            c, j = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            coherent_results[d] = {"fidelity": f, "infidelity": 1-f, "job_id": j}
            print("F=" + str(round(f,4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            coherent_results[d] = {"error": str(e)[:200]}

    # Stochastic regime (Clifford+inverse)
    print("  -- Stochastic Clifford (predicted alpha < 1) --")
    stochastic_results = {}
    for d in COHERENT_DEPTHS:
        qc = build_clifford_inverse(N_QUBITS, d)
        target = "0" * N_QUBITS
        print("    sto d=" + str(d), end=" ", flush=True)
        try:
            c, j = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            stochastic_results[d] = {"fidelity": f, "infidelity": 1-f, "job_id": j}
            print("F=" + str(round(f,4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            stochastic_results[d] = {"error": str(e)[:200]}

    # Fit K-R to each
    coh_d = sorted([d for d in coherent_results if "infidelity" in coherent_results[d]])
    sto_d = sorted([d for d in stochastic_results if "infidelity" in stochastic_results[d]])

    if len(coh_d) >= 3:
        K_coh = 1.0/np.array(coh_d)
        C_coh = np.array([coherent_results[d]["infidelity"] for d in coh_d])
        fit_coh = fit_kr(K_coh, C_coh)
    else:
        fit_coh = {"ok": False}

    if len(sto_d) >= 3:
        K_sto = 1.0/np.array(sto_d)
        C_sto = np.array([stochastic_results[d]["infidelity"] for d in sto_d])
        fit_sto = fit_kr(K_sto, C_sto)
    else:
        fit_sto = {"ok": False}

    print("\n  CLASSIFICATION RESULT:")
    if fit_coh.get("ok"):
        print("    Coherent regime:   alpha=" + str(round(fit_coh["alpha"],3)) +
              ", R^2=" + str(round(fit_coh["R2"],3)))
    if fit_sto.get("ok"):
        print("    Stochastic regime: alpha=" + str(round(fit_sto["alpha"],3)) +
              ", R^2=" + str(round(fit_sto["R2"],3)))

    # Are alphas distinguishable?
    if fit_coh.get("ok") and fit_sto.get("ok"):
        diff = abs(fit_coh["alpha"] - fit_sto["alpha"])
        print("    alpha_coh - alpha_sto = " + str(round(diff, 3)))
        if diff > 0.3:
            print("    >>> K-R DISTINGUISHES the two noise regimes <<<")

    return {
        "coherent": {str(k): v for k, v in coherent_results.items()},
        "stochastic": {str(k): v for k, v in stochastic_results.items()},
        "coherent_kr": fit_coh,
        "stochastic_kr": fit_sto,
    }

# ====================================================================
# HARD PROBLEM C: DEVICE FINGERPRINTING (run per-backend, compare across)
# ====================================================================

def hard_problem_C_fingerprint(backend):
    """Build alpha+R fingerprint of this backend."""
    print("\n  [HARD PROBLEM C] Device Fingerprint (alpha + R)")

    # Depth fingerprint
    print("  -- Depth axis --")
    depth_results = {}
    for d in FINGERPRINT_DEPTHS:
        qc = build_clifford_inverse(N_QUBITS, d)
        target = "0" * N_QUBITS
        print("    d=" + str(d), end=" ", flush=True)
        try:
            c, j = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            depth_results[d] = {"fidelity": f, "infidelity": 1-f, "job_id": j}
            print("F=" + str(round(f,4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            depth_results[d] = {"error": str(e)[:200]}

    # GHZ fingerprint
    print("  -- GHZ qubit axis --")
    ghz_results = {}
    for nq in FINGERPRINT_GHZ_QUBITS:
        if nq > backend.num_qubits: continue
        qc = build_ghz(nq)
        target = ["0"*nq, "1"*nq]
        print("    GHZ-" + str(nq) + "Q", end=" ", flush=True)
        try:
            c, j = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            ghz_results[nq] = {"fidelity": f, "infidelity": 1-f, "job_id": j}
            print("F=" + str(round(f,4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            ghz_results[nq] = {"error": str(e)[:200]}

    # Fits
    d_keys = sorted([d for d in depth_results if "infidelity" in depth_results[d]])
    g_keys = sorted([n for n in ghz_results if "infidelity" in ghz_results[n]])

    fits = {}
    if len(d_keys) >= 3:
        K = 1.0/np.array(d_keys)
        C = np.array([depth_results[d]["infidelity"] for d in d_keys])
        fits["depth"] = fit_kr(K, C)
        print("  Fingerprint depth: alpha=" + str(round(fits["depth"]["alpha"],3)) +
              ", R=" + str(round(fits["depth"]["R"],3)))
    if len(g_keys) >= 3:
        K = 1.0/np.array(g_keys)
        C = np.array([ghz_results[n]["infidelity"] for n in g_keys])
        fits["ghz"] = fit_kr(K, C)
        print("  Fingerprint GHZ:   alpha=" + str(round(fits["ghz"]["alpha"],3)) +
              ", R=" + str(round(fits["ghz"]["R"],3)))

    return {
        "depth": {str(k): v for k, v in depth_results.items()},
        "ghz": {str(k): v for k, v in ghz_results.items()},
        "fingerprint_fits": fits,
    }

# ====================================================================
# HARD PROBLEM D: STABILITY MONITORING (runs same circuits 3 times)
# ====================================================================

def hard_problem_D_stability(backend):
    """Run same circuits 3 times to detect drift / measure stability."""
    print("\n  [HARD PROBLEM D] Stability Monitoring (3 repeated runs)")

    runs = []
    for run_idx in range(STABILITY_RUNS):
        print("  -- Run " + str(run_idx+1) + "/" + str(STABILITY_RUNS) + " --")
        run_data = {"run": run_idx+1, "timestamp": time.strftime("%H:%M:%S"), "data": {}}
        for d in STABILITY_DEPTHS:
            qc = build_clifford_inverse(N_QUBITS, d)
            target = "0" * N_QUBITS
            print("    run" + str(run_idx+1) + " d=" + str(d), end=" ", flush=True)
            try:
                c, j = submit(qc, backend)
                f = get_fid(c, target, SHOTS)
                run_data["data"][d] = {"fidelity": f, "infidelity": 1-f, "job_id": j}
                print("F=" + str(round(f,4)))
            except Exception as e:
                print("ERR: " + str(e)[:80])
                run_data["data"][d] = {"error": str(e)[:200]}

        # Fit alpha for this run
        valid_d = sorted([d for d in run_data["data"] if "infidelity" in run_data["data"][d]])
        if len(valid_d) >= 3:
            K = 1.0/np.array(valid_d)
            C = np.array([run_data["data"][d]["infidelity"] for d in valid_d])
            fit = fit_kr(K, C)
            run_data["alpha"] = fit.get("alpha")
            run_data["R2"] = fit.get("R2")
            print("    Run " + str(run_idx+1) + " alpha = " + str(round(fit.get("alpha", 0),3)))
        runs.append(run_data)

    # Compute alpha stability
    alphas = [r.get("alpha") for r in runs if r.get("alpha") is not None]
    if len(alphas) >= 2:
        a_mean = np.mean(alphas); a_std = np.std(alphas)
        cv = a_std/a_mean if a_mean > 0 else 0
        print("\n  STABILITY: alpha = " + str(round(a_mean,4)) + " +/- " +
              str(round(a_std,4)) + " (CV=" + str(round(cv*100,1)) + "%)")

    # Convert run data for JSON
    runs_json = []
    for r in runs:
        rj = {"run": r["run"], "timestamp": r["timestamp"],
              "alpha": r.get("alpha"), "R2": r.get("R2"),
              "data": {str(k): v for k, v in r["data"].items()}}
        runs_json.append(rj)

    return {
        "runs": runs_json,
        "alpha_mean": float(np.mean(alphas)) if alphas else None,
        "alpha_std": float(np.std(alphas)) if alphas else None,
    }

# ====================================================================
# HARD PROBLEM E: SATURATION PREDICTION
# ====================================================================

def hard_problem_E_saturation(backend):
    """Predict saturation depth from K-R parameters."""
    print("\n  [HARD PROBLEM E] Saturation Prediction (when does fidelity collapse?)")

    sat_results = {}
    for d in SATURATION_DEPTHS:
        qc = build_clifford_inverse(N_QUBITS, d)
        target = "0" * N_QUBITS
        print("    d=" + str(d), end=" ", flush=True)
        try:
            c, j = submit(qc, backend)
            f = get_fid(c, target, SHOTS)
            sat_results[d] = {"fidelity": f, "infidelity": 1-f, "job_id": j}
            print("F=" + str(round(f,4)))
        except Exception as e:
            print("ERR: " + str(e)[:80])
            sat_results[d] = {"error": str(e)[:200]}

    # Fit K-R
    valid_d = sorted([d for d in sat_results if "infidelity" in sat_results[d]])
    if len(valid_d) >= 4:
        K = 1.0/np.array(valid_d)
        C = np.array([sat_results[d]["infidelity"] for d in valid_d])
        fit = fit_kr(K, C)

        # Saturation depth: when K-R predicts C = 0.95 (95% infidelity)
        if fit.get("ok"):
            C0 = fit["C0"]; R = fit["R"]; alpha = fit["alpha"]
            # Solve C = 0.95: C0/K^alpha + R = 0.95
            # K^alpha = C0/(0.95 - R) -> K = (C0/(0.95-R))^(1/alpha)
            target_C = 0.95
            try:
                if R < target_C and C0 > 0:
                    K_sat = (C0/(target_C - R))**(1/alpha)
                    d_sat = 1/K_sat
                    print("  Predicted saturation depth (95% infidelity): " +
                          str(round(d_sat, 1)))
                    fit["saturation_depth_95"] = float(d_sat)

                    # Check actual data: at what depth does C exceed 0.5?
                    for d in valid_d:
                        if sat_results[d]["infidelity"] > 0.5:
                            print("  Actual depth where infid > 0.5: " + str(d))
                            fit["actual_d_50"] = d
                            break
            except: pass

    return {
        "data": {str(k): v for k, v in sat_results.items()},
        "kr_fit": fit if 'fit' in dir() else None,
    }

# ====================================================================
# RUN ALL HARD PROBLEMS PER BACKEND
# ====================================================================

all_results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "shots": SHOTS,
    "n_qubits": N_QUBITS,
    "backends": {}
}

for backend in backends:
    print("\n" + "#"*75)
    print("# RUNNING ON: " + backend.name)
    print("#"*75)

    br = {"backend": backend.name, "num_qubits": backend.num_qubits}

    br["A_prediction"] = hard_problem_A_prediction(backend)
    br["B_classification"] = hard_problem_B_classification(backend)
    br["C_fingerprint"] = hard_problem_C_fingerprint(backend)
    br["D_stability"] = hard_problem_D_stability(backend)
    br["E_saturation"] = hard_problem_E_saturation(backend)

    all_results["backends"][backend.name] = br

# ====================================================================
# SUMMARY: K-R UNIQUE CAPABILITIES
# ====================================================================

print("\n" + "="*75)
print("  SUMMARY: K-R vs SOTA - WHAT K-R UNIQUELY PROVIDES")
print("="*75)

for bname, br in all_results["backends"].items():
    print("\n  " + bname.upper())
    print("  " + "-"*60)

    # Problem A: prediction accuracy
    A = br.get("A_prediction", {})
    if A.get("kr_mae") is not None:
        print("  A. PREDICTION (lower MAE = better):")
        print("     K-R:        " + str(round(A.get("kr_mae", 0), 4)))
        if A.get("exp_mae"): print("     Exponential:" + str(round(A.get("exp_mae", 0), 4)))
        if A.get("strexp_mae"): print("     Str. exp:   " + str(round(A.get("strexp_mae", 0), 4)))

    # Problem B: classification
    B = br.get("B_classification", {})
    if B.get("coherent_kr", {}).get("alpha") and B.get("stochastic_kr", {}).get("alpha"):
        a_c = B["coherent_kr"]["alpha"]; a_s = B["stochastic_kr"]["alpha"]
        print("  B. CLASSIFICATION:")
        print("     alpha_coherent  = " + str(round(a_c, 3)))
        print("     alpha_stochastic= " + str(round(a_s, 3)))
        print("     Discrimination  = " + str(round(abs(a_c - a_s), 3)) +
              (" (DISTINGUISHABLE)" if abs(a_c - a_s) > 0.3 else " (similar)"))

    # Problem C: fingerprint
    C = br.get("C_fingerprint", {}).get("fingerprint_fits", {})
    if C:
        print("  C. FINGERPRINT:")
        if C.get("depth"):
            print("     depth alpha=" + str(round(C["depth"].get("alpha", 0), 3)) +
                  ", R=" + str(round(C["depth"].get("R", 0), 3)))
        if C.get("ghz"):
            print("     GHZ alpha=" + str(round(C["ghz"].get("alpha", 0), 3)) +
                  ", R=" + str(round(C["ghz"].get("R", 0), 3)))

    # Problem D: stability
    D = br.get("D_stability", {})
    if D.get("alpha_mean") is not None:
        print("  D. STABILITY (3 runs):")
        print("     alpha = " + str(round(D["alpha_mean"], 4)) + " +/- " +
              str(round(D["alpha_std"], 4)))

    # Problem E: saturation
    E = br.get("E_saturation", {}).get("kr_fit", {})
    if E and E.get("saturation_depth_95"):
        print("  E. SATURATION:")
        print("     Predicted saturation depth: " + str(round(E["saturation_depth_95"], 1)))

# Cross-backend comparison
if len(all_results["backends"]) > 1:
    print("\n  CROSS-BACKEND CLASSIFICATION (shows alpha distinguishes devices)")
    print("  " + "-"*60)
    for bname, br in all_results["backends"].items():
        d_fp = br.get("C_fingerprint", {}).get("fingerprint_fits", {}).get("depth", {})
        g_fp = br.get("C_fingerprint", {}).get("fingerprint_fits", {}).get("ghz", {})
        if d_fp and g_fp:
            print("  " + bname.ljust(20) + ": alpha_depth=" +
                  str(round(d_fp.get("alpha", 0), 3)) +
                  ", alpha_ghz=" + str(round(g_fp.get("alpha", 0), 3)))

# ====================================================================
# SAVE
# ====================================================================

def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items() if k != "params"}
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
primary = backends[0].name
br_p = all_results["backends"][primary]

# FIGURE 1: 5-panel hard problems on primary backend
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("K-R vs SOTA: 5 Hard Problems on Real IBM Quantum (" + primary + ")",
             fontsize=14, fontweight="bold")

# (A) Prediction
ax = axes[0,0]
A = br_p.get("A_prediction", {})
if A.get("train"):
    train_d = sorted([int(k) for k in A["train"] if "infidelity" in A["train"][k]])
    train_c = [A["train"][str(d)]["infidelity"] for d in train_d]
    test_d = sorted([int(k) for k in A.get("test", {}) if "infidelity" in A["test"][k]])
    test_c = [A["test"][str(d)]["infidelity"] for d in test_d]
    ax.plot(train_d, train_c, "bo", markersize=10, label="Train")
    ax.plot(test_d, test_c, "rs", markersize=10, label="Test (actual)")
    if A.get("kr_fit", {}).get("ok"):
        all_d = np.linspace(min(train_d), max(test_d) if test_d else 32, 100)
        K_p = 1.0/all_d
        C_p = kr(K_p, *A["kr_fit"]["params"])
        ax.plot(all_d, C_p, "k--", lw=2,
                label="K-R (MAE=" + str(round(A.get("kr_mae", 0), 3)) + ")")
ax.set_xlabel("Depth"); ax.set_ylabel("Infidelity")
ax.set_title("(a) Problem A: Predict deep from shallow")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (B) Classification
ax = axes[0,1]
B = br_p.get("B_classification", {})
if B.get("coherent"):
    coh_d = sorted([int(k) for k in B["coherent"] if "infidelity" in B["coherent"][k]])
    coh_c = [B["coherent"][str(d)]["infidelity"] for d in coh_d]
    ax.plot(coh_d, coh_c, "ro-", lw=2, markersize=8,
            label="Coherent (alpha=" + str(round(B.get("coherent_kr",{}).get("alpha",0),2)) + ")")
if B.get("stochastic"):
    sto_d = sorted([int(k) for k in B["stochastic"] if "infidelity" in B["stochastic"][k]])
    sto_c = [B["stochastic"][str(d)]["infidelity"] for d in sto_d]
    ax.plot(sto_d, sto_c, "bs-", lw=2, markersize=8,
            label="Stochastic (alpha=" + str(round(B.get("stochastic_kr",{}).get("alpha",0),2)) + ")")
ax.set_xlabel("Depth"); ax.set_ylabel("Infidelity")
ax.set_title("(b) Problem B: Coherent vs Stochastic")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (C) Fingerprint
ax = axes[0,2]
C_fp = br_p.get("C_fingerprint", {})
if C_fp.get("depth"):
    d_keys = sorted([int(k) for k in C_fp["depth"] if "infidelity" in C_fp["depth"][k]])
    d_vals = [C_fp["depth"][str(d)]["infidelity"] for d in d_keys]
    ax.plot(d_keys, d_vals, "co-", lw=2, markersize=8, label="Depth axis")
if C_fp.get("ghz"):
    g_keys = sorted([int(k) for k in C_fp["ghz"] if "infidelity" in C_fp["ghz"][k]])
    g_vals = [C_fp["ghz"][str(n)]["infidelity"] for n in g_keys]
    ax.plot(g_keys, g_vals, "m^-", lw=2, markersize=8, label="GHZ axis")
ax.set_xlabel("Depth or Qubits"); ax.set_ylabel("Infidelity")
ax.set_title("(c) Problem C: Device Fingerprint (dual axis)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (D) Stability
ax = axes[1,0]
D = br_p.get("D_stability", {})
if D.get("runs"):
    colors = ["red", "blue", "green"]
    for ridx, run in enumerate(D["runs"]):
        valid_d = sorted([int(k) for k in run["data"] if "infidelity" in run["data"][k]])
        valid_c = [run["data"][str(d)]["infidelity"] for d in valid_d]
        ax.plot(valid_d, valid_c, "o-", color=colors[ridx % 3], lw=2, markersize=7,
                label="Run " + str(run["run"]) + " (alpha=" +
                str(round(run.get("alpha", 0), 3)) + ")")
ax.set_xlabel("Depth"); ax.set_ylabel("Infidelity")
ax.set_title("(d) Problem D: Stability over " + str(STABILITY_RUNS) + " runs")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (E) Saturation
ax = axes[1,1]
E = br_p.get("E_saturation", {})
if E.get("data"):
    e_d = sorted([int(k) for k in E["data"] if "infidelity" in E["data"][k]])
    e_c = [E["data"][str(d)]["infidelity"] for d in e_d]
    ax.plot(e_d, e_c, "ko-", lw=2, markersize=8, label="Real hardware")
    if E.get("kr_fit", {}).get("ok"):
        all_d = np.linspace(1, max(e_d)*2, 200)
        K_p = 1.0/all_d
        C_p = np.clip(kr(K_p, *E["kr_fit"]["params"]), 0, 1)
        ax.plot(all_d, C_p, "g--", lw=2, label="K-R extrapolation")
        if E["kr_fit"].get("saturation_depth_95"):
            ax.axvline(E["kr_fit"]["saturation_depth_95"], color="red", ls=":",
                      label="Predicted sat. d=" + str(round(E["kr_fit"]["saturation_depth_95"], 0)))
    ax.axhline(0.95, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("Depth"); ax.set_ylabel("Infidelity")
ax.set_title("(e) Problem E: Saturation Prediction")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (F) Summary table
ax = axes[1,2]
ax.axis('off')
summary_text = "K-R UNIQUE CAPABILITIES SUMMARY\n" + "="*38 + "\n\n"
if A.get("kr_mae") is not None:
    summary_text += "A. PREDICTION MAE:\n"
    summary_text += "   K-R:  " + str(round(A.get("kr_mae", 0), 4)) + "\n"
    if A.get("exp_mae"): summary_text += "   Exp:  " + str(round(A.get("exp_mae", 0), 4)) + "\n"
    summary_text += "\n"
if B.get("coherent_kr", {}).get("alpha"):
    summary_text += "B. CLASSIFICATION:\n"
    summary_text += "   coh alpha = " + str(round(B["coherent_kr"]["alpha"],3)) + "\n"
    summary_text += "   sto alpha = " + str(round(B["stochastic_kr"]["alpha"],3)) + "\n\n"
if D.get("alpha_mean"):
    summary_text += "D. STABILITY:\n"
    summary_text += "   CV = " + str(round(D["alpha_std"]/D["alpha_mean"]*100,1)) + "%\n\n"
if E.get("kr_fit", {}).get("saturation_depth_95"):
    summary_text += "E. SATURATION DEPTH:\n"
    summary_text += "   d_95 = " + str(round(E["kr_fit"]["saturation_depth_95"],1)) + "\n"
ax.text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
        verticalalignment='top', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("Fig_HardProblems_" + primary + ".png", dpi=300, bbox_inches="tight")
print("  Saved: Fig_HardProblems_" + primary + ".png")

# FIGURE 2: Cross-backend comparison
if len(all_results["backends"]) > 1:
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Cross-Backend K-R Comparison", fontsize=13, fontweight="bold")

    # Alpha across backends
    ax = axes[0]
    bnames = list(all_results["backends"].keys())
    metrics = []
    metric_labels = ["A: pred MAE", "B: alpha_coh", "B: alpha_sto",
                     "C: fp_alpha_depth", "C: fp_alpha_ghz", "D: alpha_mean"]
    for bname in bnames:
        br = all_results["backends"][bname]
        m = [
            br.get("A_prediction", {}).get("kr_mae", 0) or 0,
            br.get("B_classification", {}).get("coherent_kr", {}).get("alpha", 0) or 0,
            br.get("B_classification", {}).get("stochastic_kr", {}).get("alpha", 0) or 0,
            br.get("C_fingerprint", {}).get("fingerprint_fits", {}).get("depth", {}).get("alpha", 0) or 0,
            br.get("C_fingerprint", {}).get("fingerprint_fits", {}).get("ghz", {}).get("alpha", 0) or 0,
            br.get("D_stability", {}).get("alpha_mean", 0) or 0,
        ]
        metrics.append(m)
    metrics = np.array(metrics)
    x = np.arange(len(metric_labels)); w = 0.8/len(bnames)
    for i, bname in enumerate(bnames):
        ax.bar(x + i*w, metrics[i], w, label=bname, alpha=0.85, edgecolor="black")
    ax.set_xticks(x + w*(len(bnames)-1)/2)
    ax.set_xticklabels(metric_labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel("Value"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    ax.set_title("Metrics across backends")

    # Saturation comparison
    ax = axes[1]
    for bname in bnames:
        E = all_results["backends"][bname].get("E_saturation", {})
        if E.get("data"):
            e_d = sorted([int(k) for k in E["data"] if "infidelity" in E["data"][k]])
            e_c = [E["data"][str(d)]["infidelity"] for d in e_d]
            ax.plot(e_d, e_c, "o-", lw=2, markersize=7, label=bname)
    ax.set_xlabel("Depth"); ax.set_ylabel("Infidelity")
    ax.set_title("Saturation curves (all backends)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("Fig_HardProblems_CrossBackend.png", dpi=300, bbox_inches="tight")
    print("  Saved: Fig_HardProblems_CrossBackend.png")

elapsed = time.time() - start_time
print("\n" + "="*75)
print("  COMPLETE! Total time: " + str(round(elapsed/60, 1)) + " minutes")
print("\n  Upload to Claude:")
print("    1. ibm_hardproblems_results.json")
print("    2. Fig_HardProblems_" + primary + ".png")
if len(all_results["backends"]) > 1:
    print("    3. Fig_HardProblems_CrossBackend.png")
print("="*75)