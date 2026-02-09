#include "udf.h"
#include <stdio.h>

#define MAX_POINTS 200000

static int n_points = 0;
static real ttab[MAX_POINTS];
static real ptab[MAX_POINTS];

/* interpolazione lineare pc(t) */
static real pc_interp(real t)
{
    if (n_points < 2) return ptab[0];

    if (t <= ttab[0]) return ptab[0];
    if (t >= ttab[n_points-1]) return ptab[n_points-1];

    int i;
    for (i = 0; i < n_points-1; i++) {
        if (t >= ttab[i] && t <= ttab[i+1]) {
            real a = (t - ttab[i]) / (ttab[i+1] - ttab[i]);
            return ptab[i] + a * (ptab[i+1] - ptab[i]);
        }
    }
    return ptab[n_points-1];
}

/* lettura CSV */
DEFINE_ON_DEMAND(read_pc_table)
{
    FILE *fp = fopen("pc_vs_time_Pa.csv", "r");
    if (!fp) {
        Message("\n[UDF] ERRORE: pc_vs_time_Pa.csv non trovato\n");
        return;
    }

    char line[256];
    n_points = 0;

    /* salta header */
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp) && n_points < MAX_POINTS) {
        double t, p;
        if (sscanf(line, "%lf,%lf", &t, &p) == 2) {
            ttab[n_points] = (real)t;
            ptab[n_points] = (real)p;
            n_points++;
        }
    }

    fclose(fp);
    Message("\n[UDF] Caricati %d punti di pc(t)\n", n_points);
}

/* boundary condition */
DEFINE_PROFILE(pc_time_bc, thread, i)
{
    face_t f;
    real time = CURRENT_TIME;
    real p = pc_interp(time);

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, i) = p;
    }
    end_f_loop(f, thread)
}
