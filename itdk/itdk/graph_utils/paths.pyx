# distutils: language = c++
# distutils: sources = _paths.cpp

from libcpp.vector cimport vector


cdef extern from "_paths.h":
    vector[vector[float]] cpp_all_paths(vector[vector[float]] WU, vector[int] targets, int n_threads)

def all_paths(object g, list tgs, int n_threads):
    cdef int num_of_nodes = g.num_vertices()
    cdef vector[vector[float]] WU
    cdef vector[float] row
    cdef vector[int] targets
    edges = g.edges()
    weights = g.ep.weight

    for t in tgs:
        targets.push_back(t)
    for i in range(num_of_nodes):
        for j in range(i, num_of_nodes):
            row.push_back(0.0)
        if row.size() > 0:
            WU.push_back(row)
            row.erase(row.begin(), row.end())
    for w, (ep, es) in zip(weights, edges):
        p = int(ep)
        s = int(es)
        if p > s:
            WU[s][p - s] = float(w)
        else:
            WU[p][s - p] = float(w)

    return cpp_all_paths(WU, targets, n_threads)
