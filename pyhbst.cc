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

// a binary matchable is a keypoint and a descriptor
typedef BinaryMatchable<KeyPt, 128> Matchable128;
typedef BinaryMatchable<KeyPt, 256> Matchable256;

typedef BinaryMatch<Matchable128, double> Match128;
typedef BinaryMatch<Matchable256, double> Match256;

PYBIND11_MODULE(pyhbst, m) {
    m.doc() = "Python bindings for HBST";

    py::enum_<SplittingStrategy>(m, "SplittingStrategy")
        .value("DoNothing", SplittingStrategy::DoNothing)
        .value("SplitEven", SplittingStrategy::SplitEven)
        .value("SplitUneven", SplittingStrategy::SplitUneven)
        .value("SplitRandomUniform", SplittingStrategy::SplitRandomUniform)
        .export_values();
    // 128 bit
    py::class_<Match128>(m, "Match128")
        .def(py::init<>())
        .def(py::init<const Match128&>())
        .def_readonly("distance", &Match128::distance)
        .def_readonly("object_query", &Match128::object_query)
        .def_readonly("object_references", &Match128::object_references)
        .def_readonly("matchable_query", &Match128::matchable_query)
        .def_readonly("matchable_references", &Match128::matchable_references);

    py::class_<Matchable128>(m, "Matchable128")
        .def(py::init<KeyPt, const std::vector<bool>&, const uint64_t&>())
        .def("distance", &Matchable128::distance)
        .def("getDescriptor", &Matchable128::getDescriptorAsBoolVector)
        .def("getImageIdentifier", &Matchable128::getImageIdentifier);

    py::class_<Tree128>(m, "BinarySearchTree128")
        .def(py::init<>())
        .def(py::init<const uint64_t&>())
        .def("add", &Tree128::add)
        .def("match", &Tree128::matchWrapper)
        .def("matchAndAdd", &Tree128::matchAndAddWrapper)
        .def("trainedIdentifiers", &Tree128::trainedIdentifiers)
        .def("numberOfMatchablesUncompressed", &Tree128::numberOfMatchablesUncompressed)
        .def("numberOfMatchablesCompressed", &Tree128::numberOfMatchablesCompressed)
        .def("numberOfMergedMatchablesLastTraining", &Tree128::numberOfMergedMatchablesLastTraining)
        .def("clear", &Tree128::clear)
        .def("write", &Tree128::write)
        .def("read", &Tree128::read)
        ;

    // 256 bit
    py::class_<Match256>(m, "Match256")
        .def(py::init<>())
        .def(py::init<const Match256&>())
        .def_readonly("distance", &Match256::distance)
        .def_readonly("object_query", &Match256::object_query)
        .def_readonly("object_references", &Match256::object_references)
        .def_readonly("matchable_query", &Match256::matchable_query)
        .def_readonly("matchable_references", &Match256::matchable_references);

    py::class_<Matchable256>(m, "Matchable256")
        .def(py::init<KeyPt, const std::vector<bool>&, const uint64_t&>())
        .def("distance", &Matchable256::distance)
        .def("getDescriptor", &Matchable256::getDescriptorAsBoolVector)
        .def("getImageIdentifier", &Matchable256::getImageIdentifier);

    py::class_<Tree256>(m, "BinarySearchTree256")
        .def(py::init<>())
        .def(py::init<const uint64_t&>())
        .def("add", &Tree256::add)
        .def("match", &Tree256::matchWrapper)
        .def("matchAndAdd", &Tree256::matchAndAddWrapper)
        .def("trainedIdentifiers", &Tree256::trainedIdentifiers)
        .def("numberOfMatchablesUncompressed", &Tree256::numberOfMatchablesUncompressed)
        .def("numberOfMatchablesCompressed", &Tree256::numberOfMatchablesCompressed)
        .def("numberOfMergedMatchablesLastTraining", &Tree256::numberOfMergedMatchablesLastTraining)
        .def("clear", &Tree256::clear)
        .def("write", &Tree256::write)
        .def("read", &Tree256::read)
        ;

}
