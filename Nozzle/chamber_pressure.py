import pickle
import numpy as np
import matplotlib.pyplot as plt

PKL_PATH = "results.pkl"

OUT_CSV_PC = "pc_vs_time_Pa.csv"
OUT_CSV_TC = "Tc_vs_time_K.csv"

def as_1d_array(x, name):
    """Converte x in array 1D float se possibile."""
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
    try:
        Tc = as_1d_array(results["Tc"], "Tc")
        Tc_source = "Tc"
    except Exception as e:
        print("[DEBUG] results['Tc'] non utilizzabile:", e)

if Tc is None and "temperatures" in results and isinstance(results["temperatures"], dict):
    if "Tc" in results["temperatures"]:
        Tc = as_1d_array(results["temperatures"]["Tc"], "temperatures['Tc']")
        Tc_source = "temperatures['Tc']"

if Tc is None and "Tc_CEA" in results:
    Tc = as_1d_array(results["Tc_CEA"], "Tc_CEA")
    Tc_source = "Tc_CEA"

if Tc is None:
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

# --- plot pc (bar) ---
plt.figure()
plt.plot(t, pc / 1e5)
plt.xlabel("time [s]")
plt.ylabel("pc [bar]")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- plot Tc (K) ---
plt.figure()
plt.plot(t, Tc)
plt.xlabel("time [s]")
plt.ylabel(f"{Tc_source} [K]")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- export CSV pc ---
with open(OUT_CSV_PC, "w", encoding="utf-8") as f:
    f.write("time_s,pc_Pa\n")
    for ti, pi in zip(t, pc):
        f.write(f"{ti:.9f},{pi:.9f}\n")
print(f"[OK] Scritto: {OUT_CSV_PC} (N={len(t)})")

# --- export CSV Tc ---
with open(OUT_CSV_TC, "w", encoding="utf-8") as f:
    f.write("time_s,Tc_K\n")
    for ti, Ti in zip(t, Tc):
        f.write(f"{ti:.9f},{Ti:.9f}\n")
print(f"[OK] Scritto: {OUT_CSV_TC} (N={len(t)}, sorgente={Tc_source})")
