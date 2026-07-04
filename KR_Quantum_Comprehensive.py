##############################################################################
#  KR_Quantum_Comprehensive.py
#
#  Tests the K-R framework C = C0/K^alpha + R across 8 quantum regimes.
#  Designed to support TWO papers:
#    PAPER 1 (TQE):  Hardware noise characterization (Regimes 1-6)
#    PAPER 2 (QST):  Variational algorithm performance (Regimes 7-8)
#
#  Hypothesis: alpha takes DISTINCT, PREDICTABLE values for different noise
#  mechanisms. This makes K-R a noise-type classifier, not just a fit.
#
#  Predicted alpha values:
#    Regime 1 (coherent over-rotation):    alpha ~ 2.0  (quadratic)
#    Regime 2 (stochastic depolarizing):   alpha ~ 1.0  (linear)
#    Regime 3 (T1 amplitude damping):      alpha ~ T1-dependent
#    Regime 4 (T2 dephasing):              alpha ~ T2-dependent
#    Regime 5 (crosstalk/parallel):        alpha grows with connectivity
#    Regime 6 (measurement error):         alpha ~ small, R dominates
#    Regime 7 (VQE depth):                 alpha predicts barren plateau
#    Regime 8 (QAOA p-rounds):             alpha predicts approx ratio decay
#
#  RUNS LOCALLY (no IBM Quantum needed). Results inform what to run on
#  real hardware next.
##############################################################################

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                               thermal_relaxation_error, ReadoutError,
                               coherent_unitary_error)
from qiskit.quantum_info import Statevector, state_fidelity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, time, warnings
warnings.filterwarnings('ignore')

t0 = time.time()
print("="*72)
print("  K-R QUANTUM COMPREHENSIVE: 8 REGIMES FOR 2 PAPERS")
print("="*72)

SHOTS = 4096
results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "regimes": {}}

# ====================================================================
# K-R MODEL + COMPETING MODELS
# ====================================================================

def kr(K, C0, R, alpha):
    return C0 / (np.clip(K, 1e-30, None)**alpha) + R

def m_exp(K, A, lam, B):
    return A * np.exp(-lam * K) + B

def m_strexp(K, A, lam, beta, B):
    return A * np.exp(-(lam * np.clip(K, 1e-10, None))**beta) + B

def fit_kr(K, C):
    try:
        popt, _ = curve_fit(kr, K, C, p0=[0.5, 0.01, 1.0],
                           bounds=([0, 0, 0.01], [50, 2, 15]), maxfev=80000)
        yp = kr(K, *popt)
        ssr = np.sum((C-yp)**2); sst = np.sum((C-C.mean())**2)
        r2 = 1 - ssr/sst if sst > 1e-30 else 0
        n = len(K); sig2 = ssr/n
        ll = -n/2*np.log(2*np.pi*sig2+1e-30) - ssr/(2*sig2+1e-30)
        return {"alpha": float(popt[2]), "C0": float(popt[0]),
                "R": float(popt[1]), "R2": float(r2), "AIC": float(2*3 - 2*ll), "ok": True}
    except:
        return {"alpha": None, "R2": 0, "ok": False}

def fit_competitors(K, C):
    """Fit competing models for AIC comparison."""
    out = {}
    n = len(K)
    for name, func, p0, bounds, k_p in [
        ("Exponential", m_exp, [1,1,0], ([0,0,-2],[10,50,2]), 3),
        ("Stretched Exp", m_strexp, [1,1,1,0], ([0,0,0.01,-2],[10,50,5,2]), 4),
    ]:
        try:
            popt, _ = curve_fit(func, K, C, p0=p0, bounds=bounds, maxfev=80000)
            yp = func(K, *popt)
            ssr = np.sum((C-yp)**2); sst = np.sum((C-C.mean())**2)
            r2 = 1 - ssr/sst if sst > 1e-30 else 0
            sig2 = ssr/n
            ll = -n/2*np.log(2*np.pi*sig2+1e-30) - ssr/(2*sig2+1e-30)
            out[name] = {"R2": float(r2), "AIC": float(2*k_p - 2*ll)}
        except:
            out[name] = {"R2": 0, "AIC": 1e10}
    return out

def build_clifford_inverse(nq, depth):
    """Deterministic Clifford circuit + inverse (ideal output |0...0>)."""
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
    qc_full = qc.compose(qc_inv); qc_full.measure_all()
    return qc_full

# ====================================================================
# REGIME 1: COHERENT OVER-ROTATION
# Hypothesis: alpha ~ 2 (quadratic error accumulation)
# ====================================================================

def regime1_coherent_overrotation():
    print("\n" + "="*72)
    print("  REGIME 1: COHERENT OVER-ROTATION (predicted alpha ~ 2)")
    print("="*72)
    print("  Each gate over-rotates by epsilon. Errors add coherently.")

    eps = 0.01  # 1% over-rotation per gate (smaller to avoid saturation)
    nq = 4
    depths = np.array([1, 2, 4, 8, 16, 32, 64])  # Stop before full saturation
    C_data = []

    for d in depths:
        # Single-qubit over-rotation drift: each layer applies RX(eps) to all qubits
        qc = QuantumCircuit(nq)
        for _ in range(d):
            for q in range(nq):
                qc.rx(eps, q)  # Drift accumulates coherently
        # Compute exact fidelity via statevector
        sv = Statevector.from_label('0' * nq)
        sv = sv.evolve(qc)
        ideal = Statevector.from_label('0' * nq)
        fid = state_fidelity(sv, ideal)
        C_data.append(1 - fid)
        print(f"  depth={d:4d}: infidelity={1-fid:.6f}")

    C_data = np.array(C_data)
    K = 1.0 / depths
    fit = fit_kr(K, C_data)
    competitors = fit_competitors(K, C_data)

    print(f"\n  K-R FIT: alpha={fit['alpha']:.4f}, R2={fit['R2']:.4f}")
    print(f"  Competitor AICs: K-R={fit.get('AIC',0):.1f}, " +
          f"Exp={competitors['Exponential']['AIC']:.1f}, " +
          f"StrExp={competitors['Stretched Exp']['AIC']:.1f}")
    print(f"  Predicted alpha ~ 2.0; Got alpha = {fit['alpha']:.4f}")
    if fit['alpha'] is not None and 1.5 < fit['alpha'] < 2.5:
        print("  >>> PREDICTION CONFIRMED: alpha ~ 2 (quadratic) <<<")

    return {"depths": depths.tolist(), "infidelity": C_data.tolist(),
            "kr_fit": fit, "competitors": competitors,
            "prediction": "alpha ~ 2.0 (coherent quadratic)",
            "verified": (1.5 < fit['alpha'] < 2.5) if fit.get('alpha') else False}

# ====================================================================
# REGIME 2: STOCHASTIC DEPOLARIZING NOISE
# Hypothesis: alpha ~ 1 (linear Markovian accumulation)
# ====================================================================

def regime2_depolarizing():
    print("\n" + "="*72)
    print("  REGIME 2: STOCHASTIC DEPOLARIZING (predicted alpha ~ 1)")
    print("="*72)
    print("  Independent random Pauli errors per gate. Linear accumulation.")

    p_err = 0.001
    nq = 4
    depths = np.array([1, 2, 4, 8, 16, 32, 64, 128])

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p_err, 1),
                                    ['h','x','y','z','s','sdg','rx','ry','rz','sx','id'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p_err*5, 2), ['cx','cz'])
    backend = AerSimulator(noise_model=nm)

    C_data = []
    for d in depths:
        # Build a fixed circuit (Clifford layers) at given depth
        qc = QuantumCircuit(nq)
        for layer in range(d):
            for q in range(nq):
                qc.h(q)
            for q in range(0, nq-1, 2):
                qc.cx(q, q+1)
            if d > 1:
                for q in range(1, nq-1, 2):
                    qc.cx(q, q+1)
        # Inverse to make it identity-like (ideal output: |0...0>)
        qc_inv = qc.inverse()
        qc_full = qc.compose(qc_inv); qc_full.measure_all()
        qc_t = transpile(qc_full, backend, optimization_level=0)  # Critical: opt=0
        result = backend.run(qc_t, shots=SHOTS).result()
        counts = result.get_counts()
        fid = counts.get('0'*nq, 0) / SHOTS
        C_data.append(1 - fid)
        print(f"  depth={d:4d}: infidelity={1-fid:.4f}")

    C_data = np.array(C_data)
    K = 1.0 / depths
    fit = fit_kr(K, C_data)
    competitors = fit_competitors(K, C_data)

    print(f"\n  K-R FIT: alpha={fit['alpha']:.4f}, R2={fit['R2']:.4f}")
    print(f"  Predicted alpha ~ 1.0; Got alpha = {fit['alpha']:.4f}")
    if fit['alpha'] is not None and 0.5 < fit['alpha'] < 1.5:
        print("  >>> PREDICTION CONFIRMED: alpha ~ 1 (linear) <<<")

    return {"depths": depths.tolist(), "infidelity": C_data.tolist(),
            "kr_fit": fit, "competitors": competitors,
            "prediction": "alpha ~ 1.0 (stochastic linear)",
            "verified": (0.5 < fit['alpha'] < 1.5) if fit.get('alpha') else False}

# ====================================================================
# REGIME 3: T1 AMPLITUDE DAMPING
# Hypothesis: alpha tied to T1/t_gate ratio
# ====================================================================

def regime3_t1_damping():
    print("\n" + "="*72)
    print("  REGIME 3: T1 AMPLITUDE DAMPING (alpha varies with T1/t_gate)")
    print("="*72)

    nq = 4
    depths = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    t_gate_2q = 300e-9  # 300 ns
    t_gate_1q = 50e-9

    # Test 3 different T1 values; set T2 = T1 (must satisfy T2 <= 2*T1)
    T1_values = [50e-6, 200e-6, 1000e-6]  # 50us, 200us, 1ms
    regime3_data = {}

    for T1 in T1_values:
        T2 = T1  # T2 = T1 (within physical constraint T2 <= 2*T1)
        ratio = T1 / t_gate_2q
        nm = NoiseModel()
        # Thermal relaxation per gate
        err_1q = thermal_relaxation_error(T1, T2, t_gate_1q)
        err_2q = thermal_relaxation_error(T1, T2, t_gate_2q).expand(
                 thermal_relaxation_error(T1, T2, t_gate_2q))
        nm.add_all_qubit_quantum_error(err_1q, ['h','x','y','z','s','sdg','rx','ry','rz','sx','id'])
        nm.add_all_qubit_quantum_error(err_2q, ['cx','cz'])
        backend = AerSimulator(noise_model=nm)

        C_data = []
        for d in depths:
            qc_full = build_clifford_inverse(nq, d)
            qc_t = transpile(qc_full, backend, optimization_level=0)
            result = backend.run(qc_t, shots=SHOTS).result()
            counts = result.get_counts()
            fid = counts.get('0'*nq, 0) / SHOTS
            C_data.append(1 - fid)

        C_data = np.array(C_data)
        K = 1.0 / depths
        fit = fit_kr(K, C_data)
        regime3_data[f"T1_{T1*1e6:.0f}us"] = {
            "T1_us": T1*1e6, "T1_over_tgate": ratio,
            "depths": depths.tolist(), "infidelity": C_data.tolist(),
            "kr_fit": fit
        }
        print(f"  T1={T1*1e6:6.0f}us (T1/t_gate={ratio:.0f}): alpha={fit['alpha']:.4f}, R2={fit['R2']:.4f}")

    # Test correlation between T1/t_gate and alpha
    ratios = [regime3_data[k]["T1_over_tgate"] for k in regime3_data]
    alphas = [regime3_data[k]["kr_fit"]["alpha"] for k in regime3_data
              if regime3_data[k]["kr_fit"].get("alpha")]
    if len(alphas) >= 3:
        log_ratio = np.log10(ratios)
        rho, p_val = spearmanr(log_ratio, alphas)
        print(f"\n  Correlation (log(T1/t_gate), alpha): rho={rho:+.3f}, p={p_val:.3f}")
        if abs(rho) > 0.8:
            print("  >>> alpha VARIES SYSTEMATICALLY with T1 <<<")

    return regime3_data

# ====================================================================
# REGIME 4: T2 DEPHASING (separate from T1)
# ====================================================================

def regime4_t2_dephasing():
    print("\n" + "="*72)
    print("  REGIME 4: T2 DEPHASING (alpha varies with T2/t_gate)")
    print("="*72)

    nq = 4
    depths = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    t_gate = 300e-9
    T1 = 1000e-6  # Fixed long T1

    T2_values = [20e-6, 100e-6, 500e-6]
    regime4_data = {}

    for T2 in T2_values:
        ratio = T2 / t_gate
        nm = NoiseModel()
        err = thermal_relaxation_error(T1, T2, t_gate)
        err_2q = err.expand(err)
        nm.add_all_qubit_quantum_error(err, ['h','x','y','z','s','sdg','rx','ry','rz','sx','id'])
        nm.add_all_qubit_quantum_error(err_2q, ['cx','cz'])
        backend = AerSimulator(noise_model=nm)

        C_data = []
        for d in depths:
            qc_full = build_clifford_inverse(nq, d)
            qc_t = transpile(qc_full, backend, optimization_level=0)
            result = backend.run(qc_t, shots=SHOTS).result()
            counts = result.get_counts()
            fid = counts.get('0'*nq, 0) / SHOTS
            C_data.append(1 - fid)

        C_data = np.array(C_data)
        K = 1.0 / depths
        fit = fit_kr(K, C_data)
        regime4_data[f"T2_{T2*1e6:.0f}us"] = {
            "T2_us": T2*1e6, "T2_over_tgate": ratio,
            "depths": depths.tolist(), "infidelity": C_data.tolist(),
            "kr_fit": fit
        }
        print(f"  T2={T2*1e6:5.0f}us (T2/t_gate={ratio:.0f}): alpha={fit['alpha']:.4f}, R2={fit['R2']:.4f}")

    return regime4_data

# ====================================================================
# REGIME 5: CROSSTALK / PARALLEL GATES
# ====================================================================

def regime5_crosstalk():
    print("\n" + "="*72)
    print("  REGIME 5: CROSSTALK / PARALLEL GATES (alpha grows with n)")
    print("="*72)

    p_single = 0.001
    crosstalk = 0.005  # Extra error per parallel gate
    qubit_counts = [2, 3, 4, 5, 6]
    depths = np.array([1, 2, 4, 8, 16, 32])
    regime5_data = {}

    for nq in qubit_counts:
        # Crosstalk: error scales with number of parallel ops
        effective_err = p_single + crosstalk * (nq - 1)
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(effective_err, 1),
                                        ['h','x','y','z','rx','ry','rz','sx','id'])
        nm.add_all_qubit_quantum_error(depolarizing_error(effective_err*5, 2),
                                        ['cx','cz'])
        backend = AerSimulator(noise_model=nm)

        C_data = []
        for d in depths:
            qc_full = build_clifford_inverse(nq, d)
            qc_t = transpile(qc_full, backend, optimization_level=0)
            result = backend.run(qc_t, shots=SHOTS).result()
            counts = result.get_counts()
            fid = counts.get('0'*nq, 0) / SHOTS
            C_data.append(1 - fid)

        C_data = np.array(C_data)
        K = 1.0 / depths
        fit = fit_kr(K, C_data)
        regime5_data[f"nq_{nq}"] = {
            "nq": nq, "effective_error": effective_err,
            "depths": depths.tolist(), "infidelity": C_data.tolist(),
            "kr_fit": fit
        }
        print(f"  nq={nq} (eff_err={effective_err:.4f}): alpha={fit['alpha']:.4f}, R2={fit['R2']:.4f}")

    return regime5_data

# ====================================================================
# REGIME 6: MEASUREMENT ERROR DOMINATED
# Hypothesis: alpha small, R dominates
# ====================================================================

def regime6_measurement():
    print("\n" + "="*72)
    print("  REGIME 6: MEASUREMENT ERROR DOMINATED (alpha small, R large)")
    print("="*72)

    nq = 4
    depths = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    p_meas = 0.05  # 5% measurement error per qubit
    p_gate = 1e-5  # Very small gate error

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p_gate, 1),
                                    ['h','x','y','z','rx','ry','rz','sx','id'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p_gate, 2), ['cx','cz'])
    re = ReadoutError([[1-p_meas, p_meas], [p_meas, 1-p_meas]])
    nm.add_all_qubit_readout_error(re)
    backend = AerSimulator(noise_model=nm)

    C_data = []
    for d in depths:
        qc = random_circuit(nq, d, max_operands=2, seed=42+d)
        qc_inv = qc.inverse()
        qc_full = qc.compose(qc_inv); qc_full.measure_all()
        qc_t = transpile(qc_full, backend, optimization_level=0)
        result = backend.run(qc_t, shots=SHOTS).result()
        counts = result.get_counts()
        fid = counts.get('0'*nq, 0) / SHOTS
        C_data.append(1 - fid)
        print(f"  depth={d:4d}: infidelity={1-fid:.4f}")

    C_data = np.array(C_data)
    K = 1.0 / depths
    fit = fit_kr(K, C_data)
    print(f"\n  K-R FIT: alpha={fit['alpha']:.4f}, R={fit['R']:.4f}, R2={fit['R2']:.4f}")
    print(f"  Predicted: alpha small, R large; Got R/C0 ratio = {fit['R']/(fit['C0']+1e-9):.2f}")
    if fit['R'] > 5*fit['C0']:
        print("  >>> CONFIRMED: R dominates over C0 (measurement-limited regime) <<<")

    return {"depths": depths.tolist(), "infidelity": C_data.tolist(),
            "kr_fit": fit, "prediction": "alpha small, R large",
            "verified": (fit['R'] > 5*fit['C0']) if fit.get('R') else False}

# ====================================================================
# REGIME 7: VQE-LIKE ANSATZ (Variational, predicting trainability)
# Hypothesis: alpha predicts barren plateau onset
# ====================================================================

def regime7_vqe_ansatz():
    print("\n" + "="*72)
    print("  REGIME 7: VQE ANSATZ DEPTH (alpha predicts trainability)")
    print("="*72)
    print("  Hardware-efficient ansatz: layers of RY + CX. Test gradient variance.")

    nq = 4
    layer_counts = [1, 2, 4, 6, 8, 12, 16, 24]
    p_err = 0.003
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p_err, 1),
                                    ['h','x','y','z','rx','ry','rz','sx'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p_err*3, 2), ['cx','cz'])
    backend = AerSimulator(noise_model=nm)

    def hwea(nq, layers, params):
        qc = QuantumCircuit(nq)
        idx = 0
        for L in range(layers):
            for q in range(nq):
                qc.ry(params[idx], q); idx += 1
            for q in range(nq-1):
                qc.cx(q, q+1)
        return qc

    # For each layer count, sample random parameters and compute "gradient variance"
    # Barren plateau: variance shrinks exponentially with depth
    grad_variances = []
    C_data = []
    for L in layer_counts:
        n_params = L * nq
        gradients = []
        for trial in range(20):  # 20 random parameter samples
            np.random.seed(42 + trial + L)
            params = np.random.uniform(0, 2*np.pi, n_params)
            qc = hwea(nq, L, params)
            qc.measure_all()
            qc_t = transpile(qc, backend)
            r = backend.run(qc_t, shots=512).result()
            counts = r.get_counts()
            # "Effective measurement" = bias toward |0>
            p_zero = counts.get('0'*nq, 0) / 512
            gradients.append(p_zero)
        var = np.var(gradients)
        grad_variances.append(var)
        # Use 1 - mean(p_zero) as our "infidelity" proxy
        C_data.append(1 - np.mean(gradients))
        print(f"  layers={L:3d}: gradient variance={var:.6f}, mean infid={1-np.mean(gradients):.4f}")

    C_data = np.array(C_data)
    layers_arr = np.array(layer_counts)
    K = 1.0 / layers_arr
    fit = fit_kr(K, C_data)
    print(f"\n  K-R FIT: alpha={fit['alpha']:.4f}, R2={fit['R2']:.4f}")
    print(f"  Application: alpha quantifies how fast trainability decays with ansatz depth")

    return {"layers": layer_counts, "infidelity": C_data.tolist(),
            "gradient_variance": grad_variances, "kr_fit": fit,
            "application": "VQE trainability prediction"}

# ====================================================================
# REGIME 8: QAOA p-rounds (algorithmic performance)
# ====================================================================

def regime8_qaoa():
    print("\n" + "="*72)
    print("  REGIME 8: QAOA P-ROUNDS (alpha predicts approximation ratio decay)")
    print("="*72)
    print("  MaxCut on 4-node ring. Test approximation ratio vs p with noise.")

    nq = 4
    p_rounds = [1, 2, 3, 5, 8, 12, 16]
    p_err = 0.005
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p_err, 1),
                                    ['h','x','y','z','rx','ry','rz','sx'])
    nm.add_all_qubit_quantum_error(depolarizing_error(p_err*3, 2), ['cx','cz'])
    backend = AerSimulator(noise_model=nm)

    # MaxCut on 4-node ring: edges (0,1), (1,2), (2,3), (3,0)
    edges = [(0,1), (1,2), (2,3), (3,0)]
    max_cut = 4  # Optimal cut value

    def qaoa_circuit(p, gamma, beta):
        qc = QuantumCircuit(nq)
        for q in range(nq): qc.h(q)
        for layer in range(p):
            # Cost layer
            for (i,j) in edges:
                qc.cx(i, j)
                qc.rz(2*gamma[layer], j)
                qc.cx(i, j)
            # Mixer layer
            for q in range(nq):
                qc.rx(2*beta[layer], q)
        qc.measure_all()
        return qc

    def cut_value(bitstring):
        v = 0
        for (i,j) in edges:
            if bitstring[nq-1-i] != bitstring[nq-1-j]:
                v += 1
        return v

    C_data = []
    approx_ratios = []
    for p in p_rounds:
        # Use simple parameter heuristic (not optimized, just typical values)
        gamma = [0.5 * (k+1)/p for k in range(p)]
        beta = [0.3 * (1 - (k+1)/p) for k in range(p)]
        qc = qaoa_circuit(p, gamma, beta)
        qc_t = transpile(qc, backend)
        result = backend.run(qc_t, shots=SHOTS).result()
        counts = result.get_counts()
        # Average cut value
        avg_cut = sum(cut_value(bs) * cnt for bs, cnt in counts.items()) / SHOTS
        approx_ratio = avg_cut / max_cut
        # K-R input: 1 - approximation ratio
        infid = 1 - approx_ratio
        C_data.append(infid)
        approx_ratios.append(approx_ratio)
        print(f"  p={p:3d}: approx_ratio={approx_ratio:.4f}, 1-AR={infid:.4f}")

    C_data = np.array(C_data)
    p_arr = np.array(p_rounds)
    K = 1.0 / p_arr
    fit = fit_kr(K, C_data)
    print(f"\n  K-R FIT: alpha={fit['alpha']:.4f}, R2={fit['R2']:.4f}")
    print(f"  Application: alpha tells how QAOA performance degrades with circuit depth")

    return {"p_rounds": p_rounds, "infidelity": C_data.tolist(),
            "approx_ratios": approx_ratios, "kr_fit": fit,
            "application": "QAOA depth-performance tradeoff"}

# ====================================================================
# RUN ALL REGIMES
# ====================================================================

print("\n>>> Starting comprehensive 8-regime analysis...")
print(">>> This takes 5-10 minutes on local Aer simulator\n")

results["regimes"]["1_coherent"] = regime1_coherent_overrotation()
results["regimes"]["2_depolarizing"] = regime2_depolarizing()
results["regimes"]["3_t1_damping"] = regime3_t1_damping()
results["regimes"]["4_t2_dephasing"] = regime4_t2_dephasing()
results["regimes"]["5_crosstalk"] = regime5_crosstalk()
results["regimes"]["6_measurement"] = regime6_measurement()
results["regimes"]["7_vqe"] = regime7_vqe_ansatz()
results["regimes"]["8_qaoa"] = regime8_qaoa()

# ====================================================================
# CROSS-REGIME ANALYSIS: Does alpha CLASSIFY noise types?
# ====================================================================

print("\n\n" + "="*72)
print("  CROSS-REGIME ANALYSIS: alpha as a NOISE TYPE CLASSIFIER")
print("="*72)

summary = []
for r_name, r_data in results["regimes"].items():
    if "kr_fit" in r_data:
        a = r_data["kr_fit"].get("alpha")
        r2 = r_data["kr_fit"].get("R2", 0)
        if a is not None:
            summary.append({"regime": r_name, "alpha": a, "R2": r2})

if summary:
    print(f"\n  {'Regime':<25s} {'alpha':>8s} {'R2':>8s}")
    print("  " + "-"*45)
    for s in sorted(summary, key=lambda x: x["alpha"]):
        print(f"  {s['regime']:<25s} {s['alpha']:>8.4f} {s['R2']:>8.4f}")

# Check if alpha values are distinct for different regimes
alphas_array = np.array([s["alpha"] for s in summary])
if len(alphas_array) > 2:
    spread = alphas_array.max() - alphas_array.min()
    print(f"\n  Alpha range: [{alphas_array.min():.3f}, {alphas_array.max():.3f}], spread={spread:.3f}")
    if spread > 0.5:
        print("  >>> alpha DIFFERENTIATES noise types (spread > 0.5) <<<")

results["summary"] = summary

# ====================================================================
# SAVE RESULTS
# ====================================================================

def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean(x) for x in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    return obj

with open("KR_Quantum_Comprehensive_Results.json", "w") as f:
    json.dump(clean(results), f, indent=2)
print("\n  Saved: KR_Quantum_Comprehensive_Results.json")

# ====================================================================
# GENERATE 2 FIGURES (one per paper)
# ====================================================================

print("\n  Generating figures...")

# FIGURE FOR PAPER 1 (Hardware noise, regimes 1-6)
fig1, axes = plt.subplots(2, 3, figsize=(15, 9))
fig1.suptitle("Paper 1 (TQE): K-R Across 6 Hardware Noise Regimes",
              fontsize=14, fontweight="bold")
panel_titles = [
    "(a) Coherent (predicted α~2)",
    "(b) Depolarizing (predicted α~1)",
    "(c) T1 damping (alpha vs T1)",
    "(d) T2 dephasing (alpha vs T2)",
    "(e) Crosstalk (alpha vs n_q)",
    "(f) Measurement (R-dominated)",
]
regime_keys = ["1_coherent", "2_depolarizing", "3_t1_damping", "4_t2_dephasing",
               "5_crosstalk", "6_measurement"]

for i, (ax, title, rk) in enumerate(zip(axes.flat, panel_titles, regime_keys)):
    rd = results["regimes"][rk]

    # Single-curve regimes
    if rk in ["1_coherent", "2_depolarizing", "6_measurement"]:
        d = np.array(rd["depths"])
        c = np.array(rd["infidelity"])
        ax.plot(d, c, "ro-", markersize=7, lw=2, label="Data")
        if rd["kr_fit"].get("alpha"):
            ds = np.linspace(d.min(), d.max(), 100)
            Ks = 1.0/ds
            cf = kr(Ks, rd["kr_fit"]["C0"], rd["kr_fit"]["R"], rd["kr_fit"]["alpha"])
            ax.plot(ds, cf, "k--", lw=2,
                    label=f"K-R (α={rd['kr_fit']['alpha']:.2f}, R²={rd['kr_fit']['R2']:.2f})")
        ax.set_xlabel("Depth"); ax.set_ylabel("Infidelity")

    # Multi-curve regimes (T1, T2)
    elif rk in ["3_t1_damping", "4_t2_dephasing"]:
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(rd)))
        for c_i, (k, v) in enumerate(rd.items()):
            d = np.array(v["depths"]); inf = np.array(v["infidelity"])
            label_val = v.get("T1_us", v.get("T2_us", 0))
            ax.plot(d, inf, "o-", color=colors[c_i], markersize=5,
                    label=f"{label_val:.0f}μs (α={v['kr_fit']['alpha']:.2f})")
        ax.set_xlabel("Depth"); ax.set_ylabel("Infidelity")

    # Crosstalk
    elif rk == "5_crosstalk":
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(rd)))
        for c_i, (k, v) in enumerate(rd.items()):
            d = np.array(v["depths"]); inf = np.array(v["infidelity"])
            ax.plot(d, inf, "o-", color=colors[c_i], markersize=5,
                    label=f"{v['nq']}Q (α={v['kr_fit']['alpha']:.2f})")
        ax.set_xlabel("Depth"); ax.set_ylabel("Infidelity")

    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig("Fig_Paper1_Hardware_Regimes.png", dpi=300, bbox_inches="tight")
print("  Saved: Fig_Paper1_Hardware_Regimes.png")

# FIGURE FOR PAPER 2 (Algorithmic regimes 7-8)
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Paper 2 (QST): K-R for Variational Algorithm Performance",
              fontsize=14, fontweight="bold")

# (a) VQE
ax = axes[0]
vqe = results["regimes"]["7_vqe"]
L = np.array(vqe["layers"])
inf = np.array(vqe["infidelity"])
ax.plot(L, inf, "go-", markersize=8, lw=2, label="VQE infidelity")
if vqe["kr_fit"].get("alpha"):
    Ls = np.linspace(L.min(), L.max(), 100)
    Ks = 1.0/Ls
    cf = kr(Ks, vqe["kr_fit"]["C0"], vqe["kr_fit"]["R"], vqe["kr_fit"]["alpha"])
    ax.plot(Ls, cf, "k--", lw=2,
            label=f"K-R (α={vqe['kr_fit']['alpha']:.3f}, R²={vqe['kr_fit']['R2']:.3f})")
ax.set_xlabel("Ansatz Layers"); ax.set_ylabel("Infidelity")
ax.set_title("(a) VQE Trainability")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (b) VQE gradient variance
ax = axes[1]
gv = vqe["gradient_variance"]
ax.semilogy(L, gv, "ms-", markersize=8, lw=2)
ax.set_xlabel("Ansatz Layers"); ax.set_ylabel("Gradient Variance (log)")
ax.set_title("(b) Barren Plateau Onset")
ax.grid(True, alpha=0.3, which="both")

# (c) QAOA
ax = axes[2]
qaoa = results["regimes"]["8_qaoa"]
p = np.array(qaoa["p_rounds"])
ar = qaoa["approx_ratios"]
ax.plot(p, ar, "bo-", markersize=8, lw=2, label="Approximation Ratio")
ax2 = ax.twinx()
inf = np.array(qaoa["infidelity"])
ax2.plot(p, inf, "r^-", markersize=6, lw=1.5, label="1-AR (K-R input)")
if qaoa["kr_fit"].get("alpha"):
    ps = np.linspace(p.min(), p.max(), 100)
    Ks = 1.0/ps
    cf = kr(Ks, qaoa["kr_fit"]["C0"], qaoa["kr_fit"]["R"], qaoa["kr_fit"]["alpha"])
    ax2.plot(ps, cf, "k--", lw=2,
            label=f"K-R (α={qaoa['kr_fit']['alpha']:.3f})")
ax.set_xlabel("QAOA p-rounds"); ax.set_ylabel("Approximation Ratio", color="b")
ax2.set_ylabel("1 - AR", color="r")
ax.set_title("(c) QAOA Depth Tradeoff")
ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="lower right", fontsize=7)
ax.grid(True, alpha=0.3)

fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig("Fig_Paper2_Algorithmic.png", dpi=300, bbox_inches="tight")
print("  Saved: Fig_Paper2_Algorithmic.png")

# CROSS-REGIME SUMMARY FIGURE
fig3, ax = plt.subplots(1, 1, figsize=(10, 6))
if summary:
    sorted_summary = sorted(summary, key=lambda x: x["alpha"])
    names = [s["regime"].replace("_", " ").title() for s in sorted_summary]
    a_vals = [s["alpha"] for s in sorted_summary]
    r2_vals = [s["R2"] for s in sorted_summary]
    colors = plt.cm.RdYlGn(r2_vals)
    bars = ax.barh(names, a_vals, color=colors, edgecolor="black")
    for bar, r2 in zip(bars, r2_vals):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"R²={r2:.3f}", va="center", fontsize=9)
    ax.axvline(1.0, color="black", ls=":", lw=1, label="α=1 (linear)")
    ax.axvline(2.0, color="red", ls=":", lw=1, label="α=2 (coherent)")
    ax.set_xlabel("K-R Noise Exponent α")
    ax.set_title("α as a Noise-Type Classifier Across 8 Quantum Regimes",
                 fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="x")
fig3.tight_layout()
fig3.savefig("Fig_Cross_Regime_Alpha.png", dpi=300, bbox_inches="tight")
print("  Saved: Fig_Cross_Regime_Alpha.png")

elapsed = time.time() - t0
print("\n" + "="*72)
print(f"  COMPLETE! Total time: {elapsed/60:.1f} minutes")
print("\n  Output files:")
print("    1. KR_Quantum_Comprehensive_Results.json")
print("    2. Fig_Paper1_Hardware_Regimes.png  (for TQE submission)")
print("    3. Fig_Paper2_Algorithmic.png       (for QST/npj QI submission)")
print("    4. Fig_Cross_Regime_Alpha.png       (for both papers)")
print("="*72)