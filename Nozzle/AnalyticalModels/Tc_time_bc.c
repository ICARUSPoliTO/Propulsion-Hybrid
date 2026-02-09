#include "udf.h"
#include <stdio.h>

#define MAX_POINTS 200000

static int nTc = 0;
static real t_Tc[MAX_POINTS];
static real Tc_tab[MAX_POINTS];

static real interp_Tc(real t)
{
    if (nTc < 1) return 300.0;
    if (nTc < 2) return Tc_tab[0];

    if (t <= t_Tc[0]) return Tc_tab[0];
    if (t >= t_Tc[nTc-1]) return Tc_tab[nTc-1];

    int i;
    for (i = 0; i < nTc - 1; i++) {
        if (t >= t_Tc[i] && t <= t_Tc[i+1]) {
            real a = (t - t_Tc[i]) / (t_Tc[i+1] - t_Tc[i]);
            return Tc_tab[i] + a * (Tc_tab[i+1] - Tc_tab[i]);
        }
    }
    return Tc_tab[nTc-1];
}

DEFINE_ON_DEMAND(read_Tc_table)
{
    FILE *fp = fopen("Tc_vs_time_K.csv", "r");
    if (!fp) {
        Message("\n[UDF] ERRORE: Tc_vs_time_K.csv non trovato\n");
        return;
    }

    char line[256];
    nTc = 0;

    /* header */
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp) && nTc < MAX_POINTS) {
        double t, T;
        if (sscanf(line, "%lf,%lf", &t, &T) == 2) {
            t_Tc[nTc]  = (real)t;
            Tc_tab[nTc] = (real)T; /* K */
            nTc++;
        }
    }

    fclose(fp);

    if (nTc > 0) {
        Message("\n[UDF] Tc(t): caricati %d punti. t=[%g, %g] s\n",
                nTc, t_Tc[0], t_Tc[nTc-1]);
    } else {
        Message("\n[UDF] ATTENZIONE: Tc(t) ha 0 punti!\n");
    }
}

DEFINE_PROFILE(Tc_time_bc, thread, i)
{
    face_t f;
    real time = CURRENT_TIME;
    real T = interp_Tc(time);

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, i) = T;
    }
    end_f_loop(f, thread)
}
