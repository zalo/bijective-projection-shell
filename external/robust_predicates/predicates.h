#ifndef PRISM_ROBUST_PREDICATES_H
#define PRISM_ROBUST_PREDICATES_H

#ifdef __cplusplus
extern "C" {
#endif

void exactinit(void);
double orient2d(const double *pa, const double *pb, const double *pc);
double orient3d(const double *pa, const double *pb, const double *pc,
                const double *pd);
double incircle(const double *pa, const double *pb, const double *pc,
                const double *pd);
double insphere(const double *pa, const double *pb, const double *pc,
                const double *pd, const double *pe);

#ifdef __cplusplus
}
#endif

#endif
