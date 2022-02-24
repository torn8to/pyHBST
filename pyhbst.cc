#include "extern/pybind11/include/pybind11/pybind11.h"
#include "extern/pybind11/include/pybind11/stl.h"

#include "srrg_hbst/types/binary_match.hpp"
#include "srrg_hbst/types/binary_matchable.hpp"
#include "srrg_hbst/types/binary_node.hpp"
#include "srrg_hbst/types/binary_tree.hpp"

namespace py = pybind11;

using namespace srrg_hbst;

using DescriptorBool = std::vector<bool>;

typedef std::array<float, 2> KeyPt;

typedef BinaryTree64<KeyPt> Tree64;
typedef BinaryTree128<KeyPt> Tree128;
typedef BinaryTree256<KeyPt> Tree256;
typedef BinaryTree512<KeyPt> Tree512;

// a binary matchable is a keypoint and a descriptor
typedef BinaryMatchable<KeyPt, 64> Matchable64;
typedef BinaryMatchable<KeyPt, 128> Matchable128;
typedef BinaryMatchable<KeyPt, 256> Matchable256;
typedef BinaryMatchable<KeyPt, 512> Matchable512;

typedef BinaryMatch<Matchable64, double> Match64;
typedef BinaryMatch<Matchable128, double> Match128;
typedef BinaryMatch<Matchable256, double> Match256;
typedef BinaryMatch<Matchable512, double> Match512;

PYBIND11_MODULE(pyhbst, m) {
  m.doc() = "Python bindings for HBST";

  py::class_<Score>(m, "Score")
      .def(py::init<>())
      .def_readonly("number_of_matches", &Score::number_of_matches)
      .def_readonly("matching_ratio", &Score::matching_ratio)
      .def_readonly("identifier_reference", &Score::identifier_reference);

  py::enum_<SplittingStrategy>(m, "SplittingStrategy")
      .value("DoNothing", SplittingStrategy::DoNothing)
      .value("SplitEven", SplittingStrategy::SplitEven)
      .value("SplitUneven", SplittingStrategy::SplitUneven)
      .value("SplitRandomUniform", SplittingStrategy::SplitRandomUniform)
      .export_values();

  // 64 bit
  py::class_<Match64>(m, "Match64")
      .def(py::init<>())
      .def(py::init<const Match64&>())
      .def_readonly("distance", &Match64::distance)
      .def_readonly("object_query", &Match64::object_query)
      .def_readonly("object_references", &Match64::object_references)
      .def_readonly("matchable_query", &Match64::matchable_query)
      .def_readonly("matchable_references", &Match64::matchable_references);

  py::class_<Matchable64>(m, "Matchable64")
      .def(py::init<KeyPt, const std::vector<bool>&, const uint64_t&>())
      .def(py::init<KeyPt, const std::vector<uint8_t>&, const uint64_t&>())
      .def("distance", &Matchable64::distance)
      .def("getDescriptor", &Matchable64::getDescriptorAsBoolVector)
      .def("getImageIdentifier", &Matchable64::getImageIdentifier);

  py::class_<Tree64>(m, "BinarySearchTree64")
      .def(py::init<>())
      .def(py::init<const uint64_t&>())
      .def("add", &Tree64::addWrapper<bool>)
      .def("add", &Tree64::addWrapper<uint8_t>)
      .def("match", &Tree64::matchWrapper<bool>)
      .def("match", &Tree64::matchWrapper<uint8_t>)
      .def("matchAndAdd", &Tree64::matchAndAddWrapper<bool>)
      .def("matchAndAdd", &Tree64::matchAndAddWrapper<uint8_t>)
      .def("getScorePerImage", &Tree64::getScorePerImageWrapper<bool>)
      .def("getScorePerImage", &Tree64::getScorePerImageWrapper<uint8_t>)      
      .def("trainedIdentifiers", &Tree64::trainedIdentifiers)
      .def("numberOfMatchablesUncompressed",
           &Tree64::numberOfMatchablesUncompressed)
      .def("numberOfMatchablesCompressed",
           &Tree64::numberOfMatchablesCompressed)
      .def("numberOfMergedMatchablesLastTraining",
           &Tree64::numberOfMergedMatchablesLastTraining)
      .def("clear", &Tree64::clear)
      .def("write", &Tree64::write)
      .def("read", &Tree64::read);

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
      .def(py::init<KeyPt, const std::vector<uint8_t>&, const uint64_t&>())
      .def("distance", &Matchable128::distance)
      .def("getDescriptor", &Matchable128::getDescriptorAsBoolVector)
      .def("getImageIdentifier", &Matchable128::getImageIdentifier);

  py::class_<Tree128>(m, "BinarySearchTree128")
      .def(py::init<>())
      .def(py::init<const uint64_t&>())
      .def("add", &Tree128::addWrapper<bool>)
      .def("add", &Tree128::addWrapper<uint8_t>)
      .def("match", &Tree128::matchWrapper<bool>)
      .def("match", &Tree128::matchWrapper<uint8_t>)
      .def("matchAndAdd", &Tree128::matchAndAddWrapper<bool>)
      .def("matchAndAdd", &Tree128::matchAndAddWrapper<uint8_t>)
      .def("getScorePerImage", &Tree128::getScorePerImageWrapper<bool>)
      .def("getScorePerImage", &Tree128::getScorePerImageWrapper<uint8_t>)   
      .def("trainedIdentifiers", &Tree128::trainedIdentifiers)
      .def("numberOfMatchablesUncompressed",
           &Tree128::numberOfMatchablesUncompressed)
      .def("numberOfMatchablesCompressed",
           &Tree128::numberOfMatchablesCompressed)
      .def("numberOfMergedMatchablesLastTraining",
           &Tree128::numberOfMergedMatchablesLastTraining)
      .def("clear", &Tree128::clear)
      .def("write", &Tree128::write)
      .def("read", &Tree128::read);

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
      .def(py::init<KeyPt, const std::vector<uint8_t>&, const uint64_t&>())
      .def("distance", &Matchable256::distance)
      .def("getDescriptor", &Matchable256::getDescriptorAsBoolVector)
      .def("getImageIdentifier", &Matchable256::getImageIdentifier);

  py::class_<Tree256>(m, "BinarySearchTree256")
      .def(py::init<>())
      .def(py::init<const uint64_t&>())
      .def("add", &Tree256::addWrapper<bool>)
      .def("add", &Tree256::addWrapper<uint8_t>)
      .def("match", &Tree256::matchWrapper<bool>)
      .def("match", &Tree256::matchWrapper<uint8_t>)
      .def("matchAndAdd", &Tree256::matchAndAddWrapper<bool>)
      .def("matchAndAdd", &Tree256::matchAndAddWrapper<uint8_t>)
      .def("getScorePerImage", &Tree256::getScorePerImageWrapper<bool>)
      .def("getScorePerImage", &Tree256::getScorePerImageWrapper<uint8_t>)  
      .def("trainedIdentifiers", &Tree256::trainedIdentifiers)
      .def("numberOfMatchablesUncompressed",
           &Tree256::numberOfMatchablesUncompressed)
      .def("numberOfMatchablesCompressed",
           &Tree256::numberOfMatchablesCompressed)
      .def("numberOfMergedMatchablesLastTraining",
           &Tree256::numberOfMergedMatchablesLastTraining)
      .def("clear", &Tree256::clear)
      .def("write", &Tree256::write)
      .def("read", &Tree256::read);

  // 512 bit
  py::class_<Match512>(m, "Match512")
      .def(py::init<>())
      .def(py::init<const Match512&>())
      .def_readonly("distance", &Match512::distance)
      .def_readonly("object_query", &Match512::object_query)
      .def_readonly("object_references", &Match512::object_references)
      .def_readonly("matchable_query", &Match512::matchable_query)
      .def_readonly("matchable_references", &Match512::matchable_references);

  py::class_<Matchable512>(m, "Matchable512")
      .def(py::init<KeyPt, const std::vector<bool>&, const uint64_t&>())
      .def(py::init<KeyPt, const std::vector<uint8_t>&, const uint64_t&>())
      .def("distance", &Matchable512::distance)
      .def("getDescriptor", &Matchable512::getDescriptorAsBoolVector)
      .def("getImageIdentifier", &Matchable512::getImageIdentifier);

  py::class_<Tree512>(m, "BinarySearchTree512")
      .def(py::init<>())
      .def(py::init<const uint64_t&>())
      .def("add", &Tree512::addWrapper<bool>)
      .def("add", &Tree512::addWrapper<uint8_t>)
      .def("match", &Tree512::matchWrapper<bool>)
      .def("match", &Tree512::matchWrapper<uint8_t>)
      .def("matchAndAdd", &Tree512::matchAndAddWrapper<bool>)
      .def("matchAndAdd", &Tree512::matchAndAddWrapper<uint8_t>)
      .def("getScorePerImage", &Tree512::getScorePerImageWrapper<bool>)
      .def("getScorePerImage", &Tree512::getScorePerImageWrapper<uint8_t>)  
      .def("trainedIdentifiers", &Tree512::trainedIdentifiers)
      .def("numberOfMatchablesUncompressed",
           &Tree512::numberOfMatchablesUncompressed)
      .def("numberOfMatchablesCompressed",
           &Tree512::numberOfMatchablesCompressed)
      .def("numberOfMergedMatchablesLastTraining",
           &Tree512::numberOfMergedMatchablesLastTraining)
      .def("clear", &Tree512::clear)
      .def("write", &Tree512::write)
      .def("read", &Tree512::read);
}
