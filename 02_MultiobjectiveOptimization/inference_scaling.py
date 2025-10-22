# src/inference_scaling.py
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

class ModelConfig:
    """Configuration for a specific model with its unique parameters."""
    def __init__(self, name, c_in, c_out, t_in, t_out,
                 mu_Lin, sigma_Lin, mu_Lout, sigma_Lout,
                 acc_mean, acc_std, default_parallel=4):
        self.name = name
        self.c_in = c_in      # $ per input token
        self.c_out = c_out    # $ per output token
        self.t_in = t_in      # sec per input token
        self.t_out = t_out    # sec per output token
        self.mu_Lin = mu_Lin
        self.sigma_Lin = sigma_Lin
        self.mu_Lout = mu_Lout
        self.sigma_Lout = sigma_Lout
        self.acc_mean = acc_mean
        self.acc_std = acc_std
        self.default_parallel = default_parallel

    def __str__(self):
        return (f"{self.name}: "
                f"Cost(${self.c_in*1e6:.2f}/${self.c_out*1e6:.2f}/M), "
                f"ACC({self.acc_mean:.3f}±{self.acc_std:.3f}), "
                f"P={self.default_parallel}")

def load_model_configs(metadata_path=".project-metadata.yaml"):
    config_file = Path(__file__).parent.parent / metadata_path
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    
    configs = {}
    for key, params in data['models'].items():
        configs[key] = ModelConfig(**params)
    return configs

MODEL_CONFIGS = load_model_configs()

# sample cost and latency constraints
C_max_default = 0.50   # $
T_max_default = 60.0   # seconds

# ===============================
# MONTE CARLO SIMULATION WITH AVERAGING
# ===============================
def simulate_mc_with_model(k, model_config, mc_trials=1000, parallel_factor=8, seed=42):
    rng = np.random.default_rng(seed)
    
    costs, times, accs = [], [], []
    
    for trial in range(mc_trials):
        
        # Draw per-inference lengths for this trial
        Lin = rng.normal(model_config.mu_Lin, model_config.sigma_Lin, size=k)
        Lin = np.clip(Lin, 1, 16 * model_config.mu_Lin)
        Lout = rng.normal(model_config.mu_Lout, model_config.sigma_Lout, size=k)
        Lout = np.clip(Lout, 1, 16 * Lin)

        # Per-inference cost and time
        trial_costs = model_config.c_in * Lin + model_config.c_out * Lout
        trial_times = model_config.t_in * Lin + model_config.t_out * Lout

        # Total cost for this trial
        total_cost = trial_costs.sum()

        # Parallel-adjusted time for this trial
        if parallel_factor > 1:
            mean_time = float(np.mean(trial_times))
            total_time = (k / parallel_factor) * mean_time
        else:
            total_time = float(np.sum(trial_times))

        # Best-of-k accuracy for this trial
        trial_accs = rng.normal(model_config.acc_mean, model_config.acc_std, size=k)
        trial_accs = np.clip(trial_accs, 0.0, 1.0)
        best_acc = float(np.max(trial_accs))

        costs.append(total_cost)
        times.append(total_time)
        accs.append(best_acc)

    # Convert to numpy arrays for statistics
    costs = np.array(costs)
    times = np.array(times)
    accs = np.array(accs)

    # Calculate statistics with confidence intervals
    def calc_stats(data):
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "ci95": (float(np.percentile(data, 2.5)), float(np.percentile(data, 97.5)))
        }

    return {
        "cost": calc_stats(costs),
        "time": calc_stats(times),
        "acc": calc_stats(accs)
    }

# ===============================
# UPDATED CONSTRAINT LOGIC - TOTAL BUDGET BASIS
# ===============================
def find_feasible_k_mc(model_config, C_max_total, T_max_total, k_max=200, mc_trials=500, parallel_factor=8, seed=42):
    """Find feasible k values using total budget constraints."""
    feasible = []
    print(f"Computing feasible region with total budget constraints...")
    print(f"  C_max total: ${C_max_total:.4f}")
    print(f"  T_max total: {T_max_total:.3f}s")
    
    for k in range(parallel_factor, k_max + 1, parallel_factor): 
        if k % 20 == 0:
            print(f"  k={k}/{k_max}")
            
        stats = simulate_mc_with_model(k, model_config, mc_trials, parallel_factor, seed + k)
        
        # Use total costs and times for feasibility check (no division by k)
        total_cost = stats["cost"]["mean"]
        total_time = stats["time"]["mean"]
        
        if total_cost <= C_max_total and total_time <= T_max_total:
            feasible.append((k, stats))
    
    return feasible

def find_accuracy_optimal_mc(model_config, C_max_total, T_max_total, acc_min=0.0, k_max=200, 
                            mc_trials=500, parallel_factor=8, seed=42):
    """Find accuracy-optimal k using total budget constraints."""
    # print(f"Finding accuracy-optimal solution with total budget constraints...")
    # print(f"  C_max total: ${C_max_total:.4f}")
    # print(f"  T_max total: {T_max_total:.3f}s")
    
    best = None
    best_acc = -1
    
    for k in range(parallel_factor, k_max + 1, parallel_factor): 
        if k % 20 == 0:
            print(f"  k={k}/{k_max}")
            
        stats = simulate_mc_with_model(k, model_config, mc_trials, parallel_factor, seed + k)
        
        # Use total costs and times for constraint checks
        total_cost = stats["cost"]["mean"]
        total_time = stats["time"]["mean"]
        
        # Check total constraints
        if (total_cost <= C_max_total and 
            total_time <= T_max_total and 
            stats["acc"]["mean"] >= acc_min):
            
            if stats["acc"]["mean"] > best_acc:
                best_acc = stats["acc"]["mean"]
                best = {
                    "k": k,
                    "total_cost": total_cost,
                    "total_time": total_time,
                    "cost_per_inference": total_cost / k,
                    "time_per_inference": total_time / k,
                    "accuracy": stats["acc"]["mean"],
                    "stats": stats,
                    "model": model_config.name
                }

    return best

def find_maximum_cube_solution_mc(model_config, C_max_total, T_max_total, acc_min=0.0, k_max=200,
                                 mc_trials=500, parallel_factor=8, seed=42):
    """Find cube-optimal solution using total budget constraints."""
    # print(f"Finding cube-optimal solution with total budget constraints...")
    
    best = None
    best_vol = -1
    
    for k in range(parallel_factor, k_max + 1, parallel_factor):  # Step by P to enforce multiples
        if k % 20 == 0:
            print(f"  k={k}/{k_max}")
            
        stats = simulate_mc_with_model(k, model_config, mc_trials, parallel_factor, seed + k)
        
        # Use total metrics for constraints
        total_cost = stats["cost"]["mean"]
        total_time = stats["time"]["mean"]
        acc_mean = stats["acc"]["mean"]
        
        # Check total constraints
        if (total_cost <= C_max_total and 
            total_time <= T_max_total and 
            acc_mean >= acc_min):
            
            # Cube volume based on total budget goodness
            gC = max(0.0, 1.0 - total_cost / C_max_total)
            gT = max(0.0, 1.0 - total_time / T_max_total)
            gA = acc_mean
            vol = gC * gT * gA
            
            if vol > best_vol:
                best_vol = vol
                best = {
                    "k": k,
                    "total_cost": total_cost,
                    "total_time": total_time,
                    "cost_per_inference": total_cost / k,
                    "time_per_inference": total_time / k,
                    "accuracy": acc_mean,
                    "cube_volume": vol,
                    "stats": stats,
                    "model": model_config.name
                }

    return best

# ===============================
# PARETO, UTOPIA, KNEE, CUBE
# ===============================
def is_dominated(point1, point2):
    # points: (C, T, A)
    C1, T1, A1 = point1
    C2, T2, A2 = point2
    non_worse = (C2 <= C1) and (T2 <= T1) and (A2 >= A1)
    strictly_better = (C2 < C1) or (T2 < T1) or (A2 > A1)
    return non_worse and strictly_better

def find_pareto_frontier_mc(model_config, C_max_total, T_max_total, k_max=200, mc_trials=1, 
                           parallel_factor=8, seed=42, acc_min=0.0):
    """Find Pareto frontier using total budget constraints."""
    # print(f"Finding Pareto frontier with total budget constraints...")
    # print(f"  P={parallel_factor} (only testing k multiples of P)")
    
    feasible_points = []
    
    # Only test multiples of parallel_factor
    k_values = range(parallel_factor, k_max + 1, parallel_factor)
    
    for k in k_values:
        if k % (20 * parallel_factor) == 0:
            print(f"  k={k}/{k_max}")
            
        stats = simulate_mc_with_model(k, model_config, mc_trials, parallel_factor, seed + k)
        
        # Use total costs and times for constraints
        total_cost = stats["cost"]["mean"]
        total_time = stats["time"]["mean"]
        acc_mean = stats["acc"]["mean"]
        
        # Check total constraints INCLUDING acc_min
        if (total_cost <= C_max_total and 
            total_time <= T_max_total and
            acc_mean >= acc_min):
            
            # Store total metrics for Pareto analysis
            feasible_points.append((
                k, 
                total_cost, 
                total_time, 
                acc_mean, 
                stats
            ))

    if not feasible_points:
        return None

    # Pareto filtering based on total values
    pareto = []
    for i, (k1, C1, T1, A1, stats1) in enumerate(feasible_points):
        dominated = False
        for j, (k2, C2, T2, A2, stats2) in enumerate(feasible_points):
            if i == j:
                continue
            if is_dominated((C1, T1, A1), (C2, T2, A2)):
                dominated = True
                break
        if not dominated:
            pareto.append((k1, C1, T1, A1, stats1))
    
    pareto.sort(key=lambda x: x[0])

    # Utopia-closest selection based on total budget normalization
    def utopia_dist(C, T, A):
        return np.linalg.norm([C / C_max_total, T / T_max_total, 1 - A])

    best_item = min(pareto, key=lambda p: utopia_dist(p[1], p[2], p[3]))
    k_best, Cb, Tb, Ab, stats_best = best_item

    return {
        "k": k_best,
        "total_cost": Cb,
        "total_time": Tb,
        "cost_per_inference": Cb / k_best,
        "time_per_inference": Tb / k_best,
        "accuracy": Ab,
        "stats": stats_best,
        "pareto_points": [(k, C, T, A) for k, C, T, A, _ in pareto],
        "pareto_count": len(pareto),
        "feasible_points": len(feasible_points),
        "distance": utopia_dist(Cb, Tb, Ab),
        "model": model_config.name
    }

def find_knee_point_curvature(pareto_points, C_max_total, T_max_total):
    """Find knee point using curvature κ(k) = |p'(k) × p''(k)| / |p'(k)|³ as per paper."""
    if pareto_points is None or len(pareto_points) < 3:
        return None
    
    # Normalize trajectory based on total constraints
    P = np.array([[C / C_max_total, T / T_max_total, A] for _, C, T, A in pareto_points])
    
    max_curvature = -1
    best_idx = 0
    
    for i in range(1, len(P)-1):
        # Compute first and second derivatives using finite differences
        p_prev, p_curr, p_next = P[i-1], P[i], P[i+1]
        dp = (p_next - p_prev) / 2  # First derivative approximation
        ddp = p_next - 2*p_curr + p_prev  # Second derivative approximation
        
        # Curvature formula
        cross_product = np.cross(dp[:2], ddp[:2]) 
        cross_magnitude = abs(cross_product)
        norm_dp = np.linalg.norm(dp)
        
        if norm_dp > 1e-10:  # Avoid division by zero
            curvature = cross_magnitude / (norm_dp ** 3)
            if curvature > max_curvature:
                max_curvature = curvature
                best_idx = i
    
    return pareto_points[best_idx]

def find_knee_point(pareto_points, C_max_total, T_max_total):
    """Find knee point using perpendicular distance method (legacy)."""
    if pareto_points is None or len(pareto_points) < 3:
        return None
    
    # Normalize based on total constraints
    P = np.array([[C / C_max_total, T / T_max_total, A] for _, C, T, A in pareto_points])
    start = P[0]
    end = P[-1]
    v = end - start
    vn = np.linalg.norm(v)
    if vn == 0:
        return pareto_points[len(pareto_points)//2]

    v_unit = v / vn
    max_d = -1
    max_i = 0
    for i in range(1, len(P)-1):
        w = P[i] - start
        proj = np.dot(w, v_unit) * v_unit
        perp = w - proj
        d = np.linalg.norm(perp)
        if d > max_d:
            max_d = d
            max_i = i
    return pareto_points[max_i]

# ===============================
# UPDATED COMPARISON WRAPPER
# ===============================
def compare_methods_mc(model_config, C_max_total, T_max_total, acc_min, k_max=200, mc_trials=1, 
                      parallel_factor=8, seed=42, use_curvature_knee=True):
    """Compare optimization methods using total budget constraints."""
    print(f"\n🔄 Running Monte Carlo comparison for {model_config.name}")
    print(f"   Total budget limits: C=${C_max_total:.4f}, T={T_max_total:.3f}s")
    print(f"   MC trials: {mc_trials}, k_max: {k_max}, P: {parallel_factor}")
    print("="*70)
    
    acc_res = find_accuracy_optimal_mc(model_config, C_max_total, T_max_total, acc_min, k_max,
                                      mc_trials, parallel_factor, seed)
    
    cube_res = find_maximum_cube_solution_mc(model_config, C_max_total, T_max_total, acc_min, k_max,
                                            mc_trials, parallel_factor, seed)
    
    pareto_res = find_pareto_frontier_mc(model_config, C_max_total, T_max_total, k_max,
                                        mc_trials, parallel_factor, seed, acc_min)
    
    knee_res = None
    if pareto_res and pareto_res.get("pareto_points"):
        # Use curvature-based knee detection as per paper
        if use_curvature_knee:
            kp = find_knee_point_curvature(pareto_res["pareto_points"], C_max_total, T_max_total)
        else:
            kp = find_knee_point(pareto_res["pareto_points"], C_max_total, T_max_total)
            
        if kp:
            k_knee, Ck, Tk, Ak = kp
            # Find stats for knee point
            knee_stats = simulate_mc_with_model(k_knee, model_config, mc_trials, 
                                               parallel_factor, seed + k_knee)
            knee_res = {
                "k": k_knee, 
                "total_cost": Ck,
                "total_time": Tk,
                "cost_per_inference": Ck / k_knee,
                "time_per_inference": Tk / k_knee,
                "accuracy": Ak,
                "stats": knee_stats,
                "type": "knee_point"
            }
    
    return acc_res, cube_res, pareto_res, knee_res

# ===============================
# UPDATED VISUALIZATION — ONE FIGURE PER PLOT
# ===============================
def update_visuals_mc(selected_model, C_max_total, T_max_total, acc_min,
                     k_max=200, mc_trials=300, parallel_factor=None, seed=42):
    """Generate three separate figures instead of one combined image."""
    model_config = MODEL_CONFIGS[selected_model]
    if parallel_factor is None:
        parallel_factor = model_config.default_parallel

    print(f"🤖 {model_config.name} | P={parallel_factor} | MC={mc_trials}")
    print(f"   Costs: ${model_config.c_in*1e6:.2f}/M in, ${model_config.c_out*1e6:.2f}/M out")
    print(f"   Times: {model_config.t_in*1e6:.0f}μs/in tok, {model_config.t_out*1e6:.0f}μs/out tok")
    print(f"   ACC ~ N({model_config.acc_mean:.3f}, {model_config.acc_std:.3f})")
    print(f"   Total budget limits: C=${C_max_total:.4f}, T={T_max_total:.3f}s, ACC_min={acc_min:.3f}")

    # Monte Carlo simulations
    print("\n📊 Generating response curves...")
    # Increased limit to show curves up to higher k values for better visualization
    ks = np.arange(1, min(k_max + 1, 10**3+1))  # Up to 1000 points for broader coverage
    Cs_mean, Ts_mean, As_mean = [], [], []
    Cs_std, Ts_std, As_std = [], [], []

    for i, k in enumerate(ks):
        if i % 20 == 0:  # Adjusted print frequency for longer runs
            print(f"   k={k}/{len(ks)}")
        stats = simulate_mc_with_model(k, model_config, mc_trials//2, parallel_factor, seed + k)

        Cs_mean.append(stats["cost"]["mean"])
        Cs_std.append(stats["cost"]["std"])
        Ts_mean.append(stats["time"]["mean"])
        Ts_std.append(stats["time"]["std"])
        As_mean.append(stats["acc"]["mean"])
        As_std.append(stats["acc"]["std"])

    Cs_mean, Ts_mean, As_mean = np.array(Cs_mean), np.array(Ts_mean), np.array(As_mean)
    Cs_std, Ts_std, As_std = np.array(Cs_std), np.array(Ts_std), np.array(As_std)

    # Run optimization methods
    acc_res, cube_res, pareto_res, knee_res = compare_methods_mc(
        model_config, C_max_total, T_max_total, acc_min, k_max, mc_trials, parallel_factor, seed
    )

    # =====================================================
    # FIGURE 1 — 3D Feasible Cube (Layered Rendering)
    # =====================================================
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")

    # Feasibility computation
    feas = (Cs_mean <= C_max_total) & (Ts_mean <= T_max_total) & (As_mean >= acc_min)
    gC = np.clip(1 - Cs_mean / C_max_total, 0, 1)
    gT = np.clip(1 - Ts_mean / T_max_total, 0, 1)
    gA = np.clip(As_mean, 0, 1)
    cube = gC * gT * gA

    # ----------------------------------------
    # LAYER 1 — Base scatter (semi-transparent)
    # ----------------------------------------
    sc = ax1.scatter(Cs_mean, Ts_mean, As_mean,
                    c=cube, cmap="RdPu", s=40,
                    alpha=0.75, edgecolors="none", zorder=1)
    plt.colorbar(sc, ax=ax1, shrink=0.6, pad=0.1, label="Cube Volume")

    # ----------------------------------------
    # LAYER 2 — MC trajectory line (visible path)
    # ----------------------------------------
    # Optional glow effect (two passes)
    ax1.plot(Cs_mean, Ts_mean, As_mean,
            color="deepskyblue", lw=7.5, alpha=0.5, zorder=2)   # soft glow
    ax1.plot(Cs_mean, Ts_mean, As_mean,
            color="blue", lw=0.9, alpha=0.5,
            label="MC trajectory", zorder=3)

    # ----------------------------------------
    # Constraint planes
    # ----------------------------------------
    Cmax, Tmax, Amin = C_max_total, T_max_total, acc_min
    Cg, Tg = np.meshgrid(np.linspace(0, Cmax, 20),
                        np.linspace(0, Tmax, 20))

    ax1.plot_surface(np.full_like(Cg, Cmax), Tg,
                    np.full_like(Cg, Amin + (1 - Amin)),
                    color="tomato", alpha=0.25, zorder=0)
    ax1.plot_surface(Cg, np.full_like(Tg, Tmax),
                    np.full_like(Cg, Amin + (1 - Amin)),
                    color="royalblue", alpha=0.25, zorder=0)
    ax1.plot_surface(Cg, Tg, np.full_like(Cg, Amin),
                    color="limegreen", alpha=0.25, zorder=0)

    # ----------------------------------------
    # Cube faces and edges
    # ----------------------------------------
    verts = [
        [0, 0, Amin], [Cmax, 0, Amin], [Cmax, Tmax, Amin], [0, Tmax, Amin],
        [0, 0, 1], [Cmax, 0, 1], [Cmax, Tmax, 1], [0, Tmax, 1]
    ]
    faces = [
        [verts[j] for j in [0, 1, 2, 3]],
        [verts[j] for j in [4, 5, 6, 7]],
        [verts[j] for j in [0, 1, 5, 4]],
        [verts[j] for j in [2, 3, 7, 6]],
        [verts[j] for j in [1, 2, 6, 5]],
        [verts[j] for j in [4, 7, 3, 0]]
    ]
    ax1.add_collection3d(Poly3DCollection(faces, color="lightgreen", alpha=0.05, zorder=0))

    edge_color, edge_thick = "black", 1.8
    for X in [0, Cmax]:
        for Y in [0, Tmax]:
            ax1.plot([X, X], [Y, Y], [Amin, 1],
                    color=edge_color, lw=edge_thick, alpha=0.9, zorder=1)
    for X in [0, Cmax]:
        for Z in [Amin, 1]:
            ax1.plot([X, X], [0, Tmax], [Z, Z],
                    color=edge_color, lw=edge_thick, alpha=0.9, zorder=1)
    for Y in [0, Tmax]:
        for Z in [Amin, 1]:
            ax1.plot([0, Cmax], [Y, Y], [Z, Z],
                    color=edge_color, lw=edge_thick, alpha=0.9, zorder=1)

    # ----------------------------------------
    # LAYER 3 — Key markers (top priority)
    # ----------------------------------------
    def mark_mc(ax, res, color, marker, label):
        if res:
            ax.scatter(res.get("total_cost", res.get("cost", 0)),
                    res.get("total_time", res.get("time", 0)),
                    res["accuracy"],
                    color=color, edgecolors="black",
                    s=140, marker=marker, linewidth=1.5,
                    label=f"{label} k={res['k']}",
                    zorder=4)  # topmost layer

    mark_mc(ax1, acc_res, "gold", "o", "Accuracy-opt")
    mark_mc(ax1, cube_res, "orange", "^", "Cube-opt")
    mark_mc(ax1, pareto_res, "red", "D", "Utopia closet")
    mark_mc(ax1, knee_res, "purple", "s", "Knee")
    
    # ----------------------------------------
    # LAYER 4 — Utopian reference point (ideal target)
    # ----------------------------------------
    utopia_point = (0, 0, 1.0)
    ax1.scatter(*utopia_point,
                color="yellow", edgecolors="black",
                s=300, marker="*",
                label="Utopia ($=0,/s =0,ACC=1)",
                zorder=5)

    # ----------------------------------------
    # Axis labels & layout
    # ----------------------------------------
    ax1.set_xlabel("Total Cost ($)")
    ax1.set_ylabel("Total Time (s)")
    ax1.set_zlabel("Accuracy")
    ax1.set_title(f"Inference Scale Optimization in Feasible Space") # 3D Feasible Cube — {model_config.name}
    ax1.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()
    # fig1.savefig(f"{model_config.name}_3D_cube.pdf", bbox_inches="tight")

    # =====================================================
    # FIGURE 2 — Accuracy vs k
    # =====================================================
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.errorbar(ks, As_mean, yerr=As_std, lw=1.2, alpha=0.25, label="Accuracy ± σ")
    ax2.scatter(ks[feas], As_mean[feas], s=12, color="green", alpha=0.7, label="Feasible")
    ax2.axhline(acc_min, ls="--", color="red", label=f"ACC_min={acc_min:.2f}")
    if acc_res: ax2.scatter([acc_res["k"]], [acc_res["accuracy"]], s=100, c="gold", edgecolors="black")
    if cube_res: ax2.scatter([cube_res["k"]], [cube_res["accuracy"]], s=80, c="orange", edgecolors="black", marker="^")
    if pareto_res: ax2.scatter([pareto_res["k"]], [pareto_res["accuracy"]], s=80, c="red", edgecolors="black", marker="D")
    if knee_res: ax2.scatter([knee_res["k"]], [knee_res["accuracy"]], s=80, c="purple", edgecolors="black", marker="s")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title("Tradeoff between ACC vs Inferences (K)")
    plt.tight_layout()
    plt.show()
    # fig2.savefig(f"{model_config.name}_Accuracy_vs_k.pdf", bbox_inches="tight")

    # =====================================================
    # FIGURE 3 — Total Cost vs k
    # =====================================================
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.errorbar(ks, Cs_mean, yerr=Cs_std, lw=1.2, alpha=0.7, label="Total Cost ± σ")
    ax3.scatter(ks[feas], Cs_mean[feas], s=12, color="lightgreen", alpha=0.7, label="Feasible")
    ax3.axhline(C_max_total, ls="--", color="red", label=f"C_max=${C_max_total:.3f}")
    if acc_res: ax3.scatter([acc_res["k"]], [acc_res.get("total_cost", 0)], s=100, c="gold", edgecolors="black")
    if cube_res: ax3.scatter([cube_res["k"]], [cube_res.get("total_cost", 0)], s=80, c="orange", edgecolors="black", marker="^")
    if pareto_res: ax3.scatter([pareto_res["k"]], [pareto_res.get("total_cost", 0)], s=80, c="red", edgecolors="black", marker="D")
    if knee_res: ax3.scatter([knee_res["k"]], [knee_res.get("total_cost", 0)], s=80, c="purple", edgecolors="black", marker="s")
    ax3.set_xlabel("k")
    ax3.set_ylabel("Total Cost ($)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title("Total Cost vs k")
    plt.tight_layout()
    plt.show()
    # fig3.savefig(f"{model_config.name}_Cost_vs_k.pdf", bbox_inches="tight")

    # =====================================================
    # Text Summary
    # =====================================================
    print("\nMONTE CARLO OPTIMIZATION RESULTS (Total Budget Constraints)")
    print("=" * 70)

    def print_res(label, res, extra=""):
        if not res:
            return
        stats = res["stats"]
        print(f"\n{label}: k={res['k']} {extra}")
        print(f"   ACC: {stats['acc']['mean']:.3f} ± {stats['acc']['std']:.3f}")
        print(f"   Total: ${res['total_cost']:.3f}, {res['total_time']:.1f}s")
        print(f"   Per-inf: ${res['cost_per_inference']:.4f}, {res['time_per_inference']:.3f}s")

    print_res("Accuracy-Optimal", acc_res)
    print_res("Cube-Optimal", cube_res, f"(vol={cube_res.get('cube_volume', 0):.3f})")
    print_res("Utopia-Closest", pareto_res, f"(dist={pareto_res.get('distance', 0):.3f})")
    print_res("Knee-Point", knee_res)

# # ===============================
# # INTERACTIVE WIDGET SETUP
# # ===============================
# C_max_total_default = 0.50   # Total budget $
# T_max_total_default = 60.0   # Total time budget seconds

# print("Inference Scaling Optimization — Monte Carlo Simulations")

# widgets.interact(
#     update_visuals_mc,
#     selected_model=widgets.Dropdown(options=list(MODEL_CONFIGS.keys()),
#                                     value='gpt5', description='Model'),
#     C_max_total=widgets.FloatSlider(min=0.01, max=1.0, step=0.01,
#                               value=C_max_total_default, description="Max Total Cost ($)"),
#     T_max_total=widgets.FloatSlider(min=60.0, max=60*60, step=1.0,
#                               value=T_max_total_default, description="Max Total Time (s)"),
#     acc_min=widgets.FloatSlider(min=0.88, max=0.99, step=0.01,
#                                 value=0.83, description="Min ACC"),
#     k_max=widgets.IntSlider(min=0, max=2**7, step=4, value=2**8, description="k_max"),
#     mc_trials=widgets.IntSlider(min=300, max=500, step=10, value=300, description="MC Trials"),
#     parallel_factor=widgets.IntSlider(min=0, max=2**7, step=4,
#                                       value=MODEL_CONFIGS['gpt5'].default_parallel,
#                                       description="Parallelism (P)")
# )