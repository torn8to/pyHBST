#include "extern/pybind11/include/pybind11/pybind11.h"
#include "extern/pybind11/include/pybind11/stl.h"

#include "srrg_hbst/types/binary_match.hpp"
#include "srrg_hbst/types/binary_matchable.hpp"
#include "srrg_hbst/types/binary_node.hpp"
#include "srrg_hbst/types/binary_tree.hpp"

#include <eigen3/Eigen/Core>

namespace py = pybind11;

using namespace srrg_hbst;

using Descriptor = std::vector<bool>;

typedef std::array<float, 2> KeyPt;

typedef BinaryTree128<KeyPt> Tree128;
typedef BinaryTree256<KeyPt> Tree256;
typedef BinaryTree512<KeyPt> Tree512;

// a binary matchable is a keypoint and a descriptor
typedef BinaryMatchable<KeyPt, 128> Matchable128;
typedef BinaryMatchable<KeyPt, 256> Matchable256;
typedef BinaryMatchable<KeyPt, 512> Matchable512;


PYBIND11_MODULE(pyhbst, m) {
    m.doc() = "Python bindings for HBST";

    py::enum_<SplittingStrategy>(m, "SplittingStrategy")
        .value("DoNothing", SplittingStrategy::DoNothing)
        .value("SplitEven", SplittingStrategy::SplitEven)
        .value("SplitUneven", SplittingStrategy::SplitUneven)
        .value("SplitRandomUniform", SplittingStrategy::SplitRandomUniform)
        .export_values();

    py::class_<Matchable256>(m, "BinaryMatchable256")
        .def(py::init<KeyPt, const std::vector<bool>&, const uint64_t&>())
        .def("distance", &Matchable256::distance);

    py::class_<Tree256>(m, "BinaryTree256")
        .def(py::init<>())
        .def(py::init<const uint64_t&>())
        .def("add", &Tree256::add)
        .def("match", &Tree256::matchWrapper)
        //.def("train", &Tree256::train)
        .def("matchAndAdd", &Tree256::matchAndAddWrapper)
        // .def("getNumberOfMatches", &Tree256::getNumberOfMatches)
        // .def("getMatchingRatio", &Tree256::getMatchingRatio)
        // .def("getNumberOfMatchesLazy", &Tree256::getNumberOfMatchesLazy)
        // .def("getMatchingRatioLazy", &Tree256::getMatchingRatioLazy)
        //.def("getNumberOfMatchesLazy", &Tree256::getNumberOfMatchesLazy)
        .def("trainedIdentifiers", &Tree256::trainedIdentifiers)
        .def("numberOfMatchablesUncompressed", &Tree256::numberOfMatchablesUncompressed)
        .def("numberOfMatchablesCompressed", &Tree256::numberOfMatchablesCompressed)
        .def("numberOfMergedMatchablesLastTraining", &Tree256::numberOfMergedMatchablesLastTraining)
        .def("clear", &Tree256::clear)
        .def("write", &Tree256::write)
        .def("read", &Tree256::read)
        ;

}
