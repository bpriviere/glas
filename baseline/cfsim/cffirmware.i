%module cffirmware

// ignore GNU specific compiler attributes
#define __attribute__(x)

%{
#include "math3d.h"
#include "pptraj.h"
#include "planner.h"
#include "controller_sjc.h"
#include "stabilizer_types.h"
#include "imu_types.h"
%}

%include "stdint.i"
%include "math3d.h"
%include "pptraj.h"
%include "planner.h"
%include "controller_sjc.h"
%include "stabilizer_types.h"
%include "imu_types.h"

// support C-style arrays
%include "carrays.i"
%array_class(float, floatArray);

%inline %{
void poly4d_set(struct poly4d *poly, int dim, int coef, float val)
{
    poly->p[dim][coef] = val;
}
float poly4d_get(struct poly4d *poly, int dim, int coef)
{
    return poly->p[dim][coef];
}
struct poly4d* pp_get_piece(struct piecewise_traj *pp, int i)
{
    return &pp->pieces[i];
}
struct poly4d* malloc_poly4d(int size)
{
    return (struct poly4d*)malloc(sizeof(struct poly4d) * size);
}
%}

%extend vec {
    %pythoncode %{
        def __repr__(self):
            return "({}, {}, {})".format(self.x, self.y, self.z)
    %}
};
