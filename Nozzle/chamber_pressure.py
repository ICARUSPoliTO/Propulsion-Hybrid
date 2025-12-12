import pickle
import numpy as np
import matplotlib.pyplot as plt

PKL_PATH = "results.pkl"

OUT_CSV_PC = "pc_vs_time_Pa.csv"
OUT_CSV_TC = "Tc_vs_time_K.csv"

# =========================
# CUT SETTINGS (TRATTO LINEARE)
# =========================
T_END = 5.0              # fine tratto [s]
USE_AFTER_PEAK = True    # start subito dopo il primo picco di pc
T_START_MANUAL = 0.65   # es. 0.65 se vuoi forzare start (metti None per auto)
RESET_TIME_TO_ZERO = True  # ri-azzera tempo nel CSV (consigliato per Fluent)

def as_1d_array(x, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' non è 1D (ndim={arr.ndim}). Tipo/shape: {type(x)} / {arr.shape}")
    return arr

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

if isinstance(data, tuple) and len(data) == 2:
    t, results = data
elif isinstance(data, tuple) and len(data) == 3:
    t, _, results = data
else:
    raise ValueError("Formato pickle inatteso")

print("\n[DEBUG] File:", PKL_PATH)
print("[DEBUG] Keys results:", list(results.keys()))

t = as_1d_array(t, "time")

# --- pc ---
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

if RESET_TIME_TO_ZERO:
    t0 = t_cut[0]
    t_cut = t_cut - t0
    print(f"[INFO] Taglio: [{t_start:.6f}, {t_end:.6f}] s, poi tempo ri-azzerato (t0={t0:.6f}s)")
else:
    print(f"[INFO] Taglio: [{t_start:.6f}, {t_end:.6f}] s (tempo assoluto)")

print(f"[INFO] Punti nel tratto estratto: N={len(t_cut)}")

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
print(f"[OK] Scritto: {OUT_CSV_PC} (N={len(t_cut)})")

with open(OUT_CSV_TC, "w", encoding="utf-8") as f:
    f.write("time_s,Tc_K\n")
    for ti, Ti in zip(t_cut, Tc_cut):
        f.write(f"{ti:.9f},{Ti:.9f}\n")
print(f"[OK] Scritto: {OUT_CSV_TC} (N={len(t_cut)}, sorgente={Tc_source})")
