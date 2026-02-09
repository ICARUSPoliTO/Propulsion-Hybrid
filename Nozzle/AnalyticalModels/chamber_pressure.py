import pickle
import numpy as np
import matplotlib.pyplot as plt

PKL_PATH = "results.pkl"

OUT_CSV_PC = "pc_vs_time_Pa.csv"
OUT_CSV_TC = "Tc_vs_time_K.csv"

# =========================
# FULL MISSION SETTINGS
# =========================
# In precedenza lo script poteva "tagliare" un sotto-intervallo del profilo
# (es. tratto lineare dopo il picco). Ora l'export usa l'intero segmento di
# missione disponibile nel results.pkl.

# Se True, il tempo nel CSV viene traslato in modo che il primo istante valga 0.
# (Consigliato per Fluent se CURRENT_TIME parte da 0.)
RESET_TIME_TO_ZERO = True

# =========================
# DESIGN POINT SETTINGS
# =========================
PC_TARGET_BAR = 27.0      # pressione di progetto [bar]

def as_1d_array(x, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' non è 1D (ndim={arr.ndim}). Tipo/shape: {type(x)} / {arr.shape}")
    return arr

def _get_series_1d(results_dict, key, n):
    """Ritorna array 1D float lungo n se presente; se scalare lo espande; se assente -> None."""
    if key not in results_dict:
        return None
    arr = np.asarray(results_dict[key], dtype=float).ravel()
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.ndim != 1:
        raise ValueError(f"'{key}' non è 1D (ndim={arr.ndim}). shape={arr.shape}")
    if len(arr) != n:
        raise ValueError(f"len({key})={len(arr)} != {n}")
    return arr

def _fmt(val, fmt):
    if val is None:
        return "NA"
    try:
        if np.isnan(val):
            return "NA"
    except Exception:
        pass
    return fmt.format(val)

# =========================
# LOAD PKL
# =========================
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

# supporta: (t, results) oppure (t, inputs, results)
inputs = {}
if isinstance(data, tuple) and len(data) == 2:
    t, results = data
elif isinstance(data, tuple) and len(data) == 3:
    t, inputs, results = data
else:
    raise ValueError("Formato pickle inatteso (atteso tuple len 2 o 3)")

if not isinstance(inputs, dict):
    inputs = {}

# =========================
# EXTRACT SERIES
# =========================
t = as_1d_array(t, "time")

if "pc" not in results:
    raise KeyError("Key 'pc' non trovata nei results")
pc = as_1d_array(results["pc"], "pc")

# --- Tc: prova in ordine Tc -> temperatures['Tc'] -> Tc_CEA ---
if "Tc" in results:
    Tc = as_1d_array(results["Tc"], "Tc")
    Tc_source = "Tc"
elif "temperatures" in results and isinstance(results["temperatures"], dict) and "Tc" in results["temperatures"]:
    Tc = as_1d_array(results["temperatures"]["Tc"], "temperatures['Tc']")
    Tc_source = "temperatures['Tc']"
elif "Tc_CEA" in results:
    Tc = as_1d_array(results["Tc_CEA"], "Tc_CEA")
    Tc_source = "Tc_CEA"
else:
    raise KeyError("Non riesco a trovare una Tc utilizzabile: provati 'Tc', temperatures['Tc'], 'Tc_CEA'")

# check lunghezze
if len(t) != len(pc):
    raise ValueError(f"len(t)={len(t)} != len(pc)={len(pc)}")
if len(t) != len(Tc):
    raise ValueError(f"len(t)={len(t)} != len(Tc)={len(Tc)} (sorgente Tc: {Tc_source})")

# ordina e rimuovi duplicati tempo
idx = np.argsort(t)
t = t[idx]
pc = pc[idx]
Tc = Tc[idx]

t_unique, uidx = np.unique(t, return_index=True)
t = t_unique
pc = pc[uidx]
Tc = Tc[uidx]

# =========================
# GEOMETRY FROM INPUTS (solo per r_t e D_chamber, senza stampe extra)
# =========================
At = inputs.get("At", None)
D_chamber = inputs.get("D_chamber", None)

r_throat_geom = None
if At is not None:
    At = float(At)  # [m^2]
    r_throat_geom = float(np.sqrt(At / np.pi))  # [m]

if D_chamber is not None:
    D_chamber = float(D_chamber)  # [m]

# =========================
# USA L'INTERO PROFILO DI MISSIONE (nessun taglio)
# =========================
t_cut_abs = t.copy()
pc_cut = pc.copy()
Tc_cut = Tc.copy()

if len(t_cut_abs) < 2:
    raise RuntimeError(f"Profilo troppo corto: ottenuti {len(t_cut_abs)} punti.")

# ---------------------------------------------------------
# DESIGN POINT: trova t* tale che pc(t*) = PC_TARGET_BAR
# e interpola Tc(t*).
# ---------------------------------------------------------
PC_TARGET_PA = PC_TARGET_BAR * 1e5

pc_min, pc_max = float(np.min(pc_cut)), float(np.max(pc_cut))
t_star_abs = None
Tc_star = None

if (pc_min <= PC_TARGET_PA <= pc_max):
    # Caso tipico: salita -> picco -> discesa. Per coerenza col "design point"
    # operativo, cerco la prima intersezione con pc_target DOPO il primo picco.
    # Se non esiste (o pc è monotona), ripiego sulla prima intersezione sull'intero profilo.
    def _first_crossing_time(t_arr, y_arr, y_target, start_idx=0):
        for k in range(start_idx, len(t_arr) - 1):
            y0 = y_arr[k] - y_target
            y1 = y_arr[k + 1] - y_target
            if y0 == 0.0:
                return float(t_arr[k])
            if y0 * y1 < 0.0:
                # Interpolazione lineare tra (t_k, y_k) e (t_{k+1}, y_{k+1})
                a = abs(y0) / (abs(y0) + abs(y1))
                return float(t_arr[k] + a * (t_arr[k + 1] - t_arr[k]))
        return None

    i_peak = int(np.argmax(pc_cut))
    t_star_abs = _first_crossing_time(t_cut_abs, pc_cut, PC_TARGET_PA, start_idx=i_peak)
    if t_star_abs is None:
        t_star_abs = _first_crossing_time(t_cut_abs, pc_cut, PC_TARGET_PA, start_idx=0)

    if t_star_abs is not None:
        Tc_star = float(np.interp(t_star_abs, t_cut_abs, Tc_cut))

# =========================
# reset tempo se richiesto (serve SOLO per CSV e plot)
# =========================
if RESET_TIME_TO_ZERO:
    t0 = t_cut_abs[0]
    t_cut = t_cut_abs - t0
else:
    t_cut = t_cut_abs

# =====================================================================
# Serie per interpolare gamma, R, Thrust al DESIGN POINT
# =====================================================================
n_all = len(t)
gamma_cut  = _get_series_1d(results, "gamma",  n_all)
thrust_cut = _get_series_1d(results, "Thrust", n_all)
MW_cut     = _get_series_1d(results, "MW",     n_all)

# R da MW (se disponibile)
R_cut = None
if MW_cut is not None:
    Ru_kmol = 8314.462618  # J/(kmol*K)
    mw_med = float(np.nanmedian(MW_cut))
    if 1.0 <= mw_med <= 1000.0:
        R_cut = Ru_kmol / MW_cut  # MW in kg/kmol
    elif 1e-4 <= mw_med <= 1.0:
        Ru_mol = 8.314462618
        R_cut = Ru_mol / MW_cut   # MW in kg/mol
    else:
        R_cut = Ru_kmol / MW_cut  # fallback

# =====================================================================
# STAMPA SOLO DESIGN POINT TABLE (come richiesto)
# =====================================================================
if t_star_abs is None:
    print("\n==================== DESIGN POINT TABLE ====================")
    print("[WARN] pc_target fuori dal range del profilo -> design point non calcolabile.")
    print("============================================================\n")
else:
    gamma_star  = float(np.interp(t_star_abs, t_cut_abs, gamma_cut))  if gamma_cut  is not None else None
    R_star      = float(np.interp(t_star_abs, t_cut_abs, R_cut))      if R_cut      is not None else None
    thrust_star = float(np.interp(t_star_abs, t_cut_abs, thrust_cut)) if thrust_cut is not None else None

    print("\n==================== DESIGN POINT TABLE ====================")
    print(" t_abs[s] | pc[bar] |  Tc[K]  | gamma |   R[J/kgK] | Thrust[N] | r_t[m]   | D_ch[m]")
    print("----------------------------------------------------------------------------------")
    print(
        f"{t_star_abs:8.6f} | "
        f"{PC_TARGET_BAR:7.3f} | "
        f"{Tc_star:7.3f} | "
        f"{_fmt(gamma_star, '{:.5f}'):>5} | "
        f"{_fmt(R_star, '{:.2f}'):>10} | "
        f"{_fmt(thrust_star, '{:.3f}'):>9} | "
        f"{_fmt(r_throat_geom, '{:.6f}'):>7} | "
        f"{_fmt(D_chamber, '{:.6f}'):>6}"
    )
    print("============================================================\n")

# =========================
# PLOTS (2 figure separate, SOLO tratto estratto)
# =========================
plt.figure()
plt.plot(t_cut, pc_cut / 1e5)
plt.xlabel("time [s]")
plt.ylabel("pc [bar]")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t_cut, Tc_cut)
plt.xlabel("time [s]")
plt.ylabel(f"{Tc_source} [K]")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# EXPORT CSV (SOLO tratto estratto)
# =========================
with open(OUT_CSV_PC, "w", encoding="utf-8") as f:
    f.write("time_s,pc_Pa\n")
    for ti, pi in zip(t_cut, pc_cut):
        f.write(f"{ti:.9f},{pi:.9f}\n")
print(f"[OK] Wrote: {OUT_CSV_PC} (N={len(t_cut)})")

with open(OUT_CSV_TC, "w", encoding="utf-8") as f:
    f.write("time_s,Tc_K\n")
    for ti, Ti in zip(t_cut, Tc_cut):
        f.write(f"{ti:.9f},{Ti:.9f}\n")
print(f"[OK] Wrote: {OUT_CSV_TC} (N={len(t_cut)}, source={Tc_source})")
