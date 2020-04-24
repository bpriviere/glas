%module nnexport
%include "stdint.i"

%typemap(in) float[ANY] (float temp[$1_dim0]) {
  int i;
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_ValueError, "Expected a sequence");
    SWIG_fail;
  }
  if (PySequence_Length($input) != $1_dim0) {
    PyErr_SetString(PyExc_ValueError, "Size mismatch. Expected $1_dim0 elements");
    SWIG_fail;
  }
  for (i = 0; i < $1_dim0; i++) {
    PyObject *o = PySequence_GetItem($input, i);
    if (PyNumber_Check(o)) {
      temp[i] = (float) PyFloat_AsDouble(o);
    } else {
      PyErr_SetString(PyExc_ValueError, "Sequence elements must be numbers");      
      SWIG_fail;
    }
  }
  $1 = temp;
}

%typemap(out) float const *nn_eval {
  int i;
  //$1, $1_dim0, $1_dim1
  $result = PyList_New(2);
  for (i = 0; i < 2; i++) {
    PyObject *o = PyFloat_FromDouble((double) $1[i]);
    PyList_SetItem($result,i,o);
  }
}

%{
#include "nn.h"
%}

%apply unsigned long long { uint64_t }

%include "nn.h"

