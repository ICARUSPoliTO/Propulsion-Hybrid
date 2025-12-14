import pickle
import numpy as np
import matplotlib.pyplot as plt

PKL_PATH = "results.pkl"

OUT_CSV_PC = "pc_vs_time_Pa.csv"
OUT_CSV_TC = "Tc_vs_time_K.csv"

# =========================
# CUT SETTINGS (TRATTO LINEARE)
# =========================
T_END = 5.0               # fine tratto [s]
USE_AFTER_PEAK = True     # start subito dopo il primo picco di pc
T_START_MANUAL = 0.65     # es. 0.65 se vuoi forzare start (metti None per auto)
RESET_TIME_TO_ZERO = True # ri-azzera tempo nel CSV (consigliato per Fluent)

# =========================
# DESIGN POINT SETTINGS
# =========================
PC_TARGET_BAR = 25.0      # pressione di progetto [bar]

# =========================
# EXTRA OUTPUT SETTINGS
# =========================
SHOW_ALL_KEYS = True      # (lasciato invariato, ma non stampiamo nulla oltre la DESIGN POINT TABLE)

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

# (DEBUG keys: lo lasciamo disponibile ma NON lo stampiamo, come richiesto)
# if SHOW_ALL_KEYS: ...

# =========================
# EXTRACT SERIES
# =========================
t = as_1d_array(t, "time")

if "pc" not in results:
    raise KeyError("Key 'pc' non trovata nei results")
pc = as_1d_array(results["pc"], "pc")

# --- Tc: prova in ordine Tc -> temperatures['Tc'] -> Tc_CEA ---
Tc = None
Tc_source = None

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
# GEOMETRY FROM INPUTS (serve solo per r_t_geom, senza stampare)
# =========================
At = None
if isinstance(inputs, dict):
    At = inputs.get("At", None)

if At is None:
    r_throat_geom = None
else:
    At = float(At)                    # [m^2]
    r_throat_geom = float(np.sqrt(At / np.pi))  # [m]

# =========================
# ESTRAI SOLO TRATTO "LINEARE" (dopo picco -> T_END)
# =========================
if T_START_MANUAL is not None:
    t_start = float(T_START_MANUAL)
else:
    if USE_AFTER_PEAK:
        i_peak = int(np.argmax(pc))
        t_start = float(t[i_peak]) + 1e-9  # subito dopo il picco
    else:
        t_start = float(t[0])

t_end = min(float(T_END), float(t[-1]))

mask = (t >= t_start) & (t <= t_end)
t_cut  = t[mask]
pc_cut = pc[mask]
Tc_cut = Tc[mask]

if len(t_cut) < 2:
    raise RuntimeError(
        f"Taglio non valido: ottenuti {len(t_cut)} punti. "
        f"Controlla t_start={t_start} e t_end={t_end}."
    )

# salva tempo assoluto del tratto (prima di eventuale reset)
t_cut_abs = t_cut.copy()

# ---------------------------------------------------------
# DESIGN POINT: trova t* tale che pc(t*) = PC_TARGET_BAR
# e interpola Tc(t*).
# ---------------------------------------------------------
PC_TARGET_PA = PC_TARGET_BAR * 1e5

pc_min, pc_max = float(np.min(pc_cut)), float(np.max(pc_cut))
t_star_abs = None
Tc_star = None

if (pc_min <= PC_TARGET_PA <= pc_max):
    # robustezza al rumore: rendi pc monotona secondo il trend medio
    slope = np.polyfit(t_cut_abs, pc_cut, 1)[0]
    pc_work = pc_cut.copy()

    if slope < 0:  # in media decrescente
        pc_work = np.maximum.accumulate(pc_work[::-1])[::-1]
        t_star_abs = float(np.interp(PC_TARGET_PA, pc_work[::-1], t_cut_abs[::-1]))
    else:          # in media crescente
        pc_work = np.minimum.accumulate(pc_work)
        t_star_abs = float(np.interp(PC_TARGET_PA, pc_work, t_cut_abs))

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
gamma_all  = _get_series_1d(results, "gamma",  n_all)
thrust_all = _get_series_1d(results, "Thrust", n_all)
MW_all     = _get_series_1d(results, "MW",     n_all)

mask_full = (t >= t_start) & (t <= t_end)
gamma_cut  = gamma_all[mask_full]  if gamma_all  is not None else None
thrust_cut = thrust_all[mask_full] if thrust_all is not None else None
MW_cut     = MW_all[mask_full]     if MW_all     is not None else None

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
    print("[WARN] pc_target fuori dal range del tratto estratto -> design point non calcolabile.")
    print("============================================================\n")
else:
    gamma_star  = float(np.interp(t_star_abs, t_cut_abs, gamma_cut))  if gamma_cut  is not None else None
    R_star      = float(np.interp(t_star_abs, t_cut_abs, R_cut))      if R_cut      is not None else None
    thrust_star = float(np.interp(t_star_abs, t_cut_abs, thrust_cut)) if thrust_cut is not None else None
    r_geom_star = float(r_throat_geom) if r_throat_geom is not None else None

    print("\n==================== DESIGN POINT TABLE ====================")
    print(" t_abs[s] | pc[bar] |  Tc[K]  | gamma |   R[J/kgK] | Thrust[N] | r_t[m]")
    print("---------------------------------------------------------------------")
    print(
        f"{t_star_abs:8.6f} | "
        f"{PC_TARGET_BAR:7.3f} | "
        f"{Tc_star:7.3f} | "
        f"{_fmt(gamma_star, '{:.5f}'):>5} | "
        f"{_fmt(R_star, '{:.2f}'):>10} | "
        f"{_fmt(thrust_star, '{:.3f}'):>9} | "
        f"{_fmt(r_geom_star, '{:.6f}'):>7}"
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
