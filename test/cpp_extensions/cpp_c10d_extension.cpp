#include "cpp_c10d_extension.hpp"

#include <map>

namespace c10d {

std::shared_ptr<ProcessGroup> ProcessGroupC10dTest::createProcessGroupC10dTest(const std::shared_ptr<::c10d::Store>& store, int rank, int size, const std::string& groupName = "") {
  return std::make_shared<ProcessGroupC10dTest>(rank, size);
}

ProcessGroupC10dTest::ProcessGroupC10dTest(int rank, int size)
    : ProcessGroup(rank, size) {}

ProcessGroupC10dTest::~ProcessGroupC10dTest() {}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  return NULL;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  return NULL;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  return NULL;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupC10dTest does not support allgather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::barrier(
    const BarrierOptions& opts) {
  throw std::runtime_error("ProcessGroupC10dTest does not support barrier");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupC10dTest does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupC10dTest does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupC10dTest does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupC10dTest does not support send");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupC10dTest does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupC10dTest does not support recvAnysource");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupC10dTest::allreduce_coalesced(
      std::vector<at::Tensor>& /* unused */,
      const AllreduceCoalescedOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupC10dTest does not support allreduce_coalesced");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ProcessGroupC10DTEST", &ProcessGroupC10dTest::createProcessGroupC10dTest);
}

} // namespace c10d
