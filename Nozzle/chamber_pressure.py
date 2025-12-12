import pickle
import numpy as np
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
PKL_PATH = "results.pkl"
OUT_CSV = "pc_vs_time_Pa.csv"   # time [s], pc [Pa]

# =========================
# LOAD PICKLE
# =========================
with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

# (time, results) oppure (time, inputs, results)
if isinstance(data, tuple) and len(data) == 2:
    t, results = data
elif isinstance(data, tuple) and len(data) == 3:
    t, _, results = data
else:
    raise ValueError(
        f"Formato pickle inatteso: type={type(data)}, "
        f"len={len(data) if isinstance(data, tuple) else 'n/a'}"
    )

# =========================
# EXTRACT pc(t)
# =========================
if "pc" not in results:
    raise KeyError(f"Key 'pc' non trovata. Keys disponibili: {list(results.keys())}")

t = np.asarray(t, dtype=float)
pc = np.asarray(results["pc"], dtype=float)   # [Pa]

if len(t) != len(pc):
    raise ValueError(f"len(t)={len(t)} != len(pc)={len(pc)}")

# Ordina per tempo crescente
idx = np.argsort(t)
t = t[idx]
pc = pc[idx]

# Rimuove eventuali tempi duplicati (Fluent/UDF preferisce tempi strettamente crescenti)
t_unique, unique_idx = np.unique(t, return_index=True)
pc_unique = pc[unique_idx]
t, pc = t_unique, pc_unique

# =========================
# PLOT (bar)
# =========================
pc_bar = pc / 1e5

plt.figure()
plt.plot(t, pc_bar)
plt.xlabel("time [s]")
plt.ylabel("pc [bar]")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# EXPORT CSV (Pa)
# =========================
# CSV semplice "time,pressure" con separatore virgola (UDF lo legge facilmente)
with open(OUT_CSV, "w", encoding="utf-8") as f:
    f.write("time_s,pc_Pa\n")
    for ti, pi in zip(t, pc):
        f.write(f"{ti:.9f},{pi:.9f}\n")

print(f"[OK] Scritto: {OUT_CSV}  (N={len(t)} punti)")
