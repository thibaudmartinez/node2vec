#if defined(_OPENMP)
#include <omp.h>
#endif

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "glib/base.cpp"
#include "snap/attr.cpp"
#include "snap/biasedrandomwalk.cpp"
#include "snap/gbase.cpp"
#include "snap/graph.cpp"
#include "snap/network.cpp"

namespace py = pybind11;

void AddNodeIfDoesNotExist(PWNet &Network, const int32_t &NodeId) {
    if (!Network->IsNode(NodeId)) {
        Network->AddNode(NodeId);
    }
}

const PWNet BuildNetworkFromEdgeList(
        const py::array_t <int32_t> &SrcNodes,
        const py::array_t <int32_t> &DestNodes,
        const py::array_t<double> &EdgeWeights,
        const bool &GraphIsDirected
) {
    py::buffer_info SrcNodesBuffer = SrcNodes.request();
    py::buffer_info DestNodesBuffer = DestNodes.request();
    py::buffer_info EdgeWeightsBuffer = EdgeWeights.request();

    if (SrcNodesBuffer.ndim != 1 || DestNodesBuffer.ndim != 1 ||
        EdgeWeightsBuffer.ndim != 1) {
        throw std::runtime_error("Input arrays must be of dimension 1.");
    }

    if (SrcNodesBuffer.size != DestNodesBuffer.size ||
        DestNodesBuffer.size != EdgeWeightsBuffer.size) {
        throw std::runtime_error("Input arrays must be of same size.");
    }

    int32_t *const SrcNodesPtr = (int32_t *) SrcNodesBuffer.ptr;
    int32_t *const DestNodesPtr = (int32_t *) DestNodesBuffer.ptr;
    double *const EdgeWeightsPtr = (double *) EdgeWeightsBuffer.ptr;

    PWNet Network = PWNet::New();

    for (size_t idx = 0; idx < SrcNodesBuffer.shape[0]; idx++) {
        const int32_t SrcNodeId = SrcNodesPtr[idx];
        const int32_t DestNodeId = DestNodesPtr[idx];
        const double EdgeWeight = EdgeWeightsPtr[idx];

        AddNodeIfDoesNotExist(Network, SrcNodeId);
        AddNodeIfDoesNotExist(Network, DestNodeId);

        Network->AddEdge(SrcNodeId, DestNodeId, EdgeWeight);
        if (!GraphIsDirected) {
            Network->AddEdge(DestNodeId, SrcNodeId, EdgeWeight);
        }
    }

    return Network;
}

py::array_t <int32_t> WalksVVToVector(TVVec<TInt, int64> &WalksVV) {
    const int64 XDim = WalksVV.GetXDim();
    const int64 YDim = WalksVV.GetYDim();

    const size_t Size = XDim * YDim;
    int32_t *const ArrayDataPtr = new int32_t[Size];

    for (int64 idx = 0; idx < XDim; idx++) {
        for (int64 idy = 0; idy < YDim; idy++) {
            ArrayDataPtr[idx * YDim + idy] = WalksVV(idx, idy);
        }
    }

    py::capsule FreeWhenDone(ArrayDataPtr, [](void *f) {
        int32_t *const ArrayDataPtr = reinterpret_cast<int32_t *const>(f);
        delete[] ArrayDataPtr;
    });

    return py::array_t<int32_t>(
            {XDim, YDim}, // shape
            {YDim * sizeof(int32_t), sizeof(int32_t)}, // C-style contiguous strides for int32
            ArrayDataPtr, // the data pointer
            FreeWhenDone  // numpy array references this parent
    );
}

py::array_t <int32_t> SimulateBiasedRandomWalks(
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &SrcNodes,
        const py::array_t<int32_t, py::array::c_style | py::array::forcecast> &DestNodes,
        const py::array_t<double, py::array::c_style | py::array::forcecast> &EdgeWeights,
        const bool &GraphIsDirected,
        const int &WalkLength,
        const int &NWalks,
        const double &ParamP,
        const double &ParamQ,
        const int &Workers,
        const bool &Verbose,
        const int &RandSeed
) {
    PWNet Network = BuildNetworkFromEdgeList(
            SrcNodes, DestNodes, EdgeWeights, GraphIsDirected
    );

#if defined(_OPENMP)
    omp_set_num_threads(Workers);
#endif

    TVVec<TInt, int64> WalksVV;
    if (RandSeed) {
        WalksVV = SimulateWalks(
            Network, WalkLength, NWalks, ParamP, ParamQ, Verbose, RandSeed
        );
    }
    else {
        WalksVV = SimulateWalks(
            Network, WalkLength, NWalks, ParamP, ParamQ, Verbose
        );
    }

    return WalksVVToVector(WalksVV);
}

PYBIND11_MODULE(cppsnap, m) {
m.def(
"simulate_biased_random_walks",
&SimulateBiasedRandomWalks,
"Simulate biased random walks.",
py::arg("src_nodes"),
py::arg("dest_nodes"),
py::arg("edge_weights"),
py::arg("graph_is_directed"),
py::arg("walk_length"),
py::arg("n_walks"),
py::arg("p"),
py::arg("q"),
py::arg("workers"),
py::arg("verbose"),
py::arg("rand_seed")
);

#ifdef VERSION_INFO
m.attr("__version__") = VERSION_INFO;
#else
m.attr("__version__") = "dev";
#endif
}
