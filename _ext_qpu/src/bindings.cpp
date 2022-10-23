#include "hmd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hamilton_product", &hamilton_product);
  m.def("hamilton_product_grad", &hamilton_product_grad);
}
