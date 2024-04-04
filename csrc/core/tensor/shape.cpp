/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    shape.cpp
 */

#include "shape.h"  // NOLINT

#include <sstream>

namespace allspark {

Shape::Shape(const std::vector<dim_t>& shape) : dim(shape) {}
Shape::Shape(const std::initializer_list<dim_t>& shape) : dim(shape) {}
Shape::Shape(const int ndim, const dim_t* val) : dim(val, val + ndim) {}

int Shape::Size() const { return static_cast<int>(dim.size()); }

int64_t Shape::Count(int start) const {
  if (dim.size()) {
    return Count(start, static_cast<int>(dim.size()));
  } else {
    return 0;
  }
}
int64_t Shape::Count(int start, int end) const {
  int64_t count = 1;
  for (int i = start; i < end; ++i) {
    count *= dim[i];
  }
  return count;
}

void Shape::Append(dim_t axis) { dim.emplace_back(axis); }

std::string Shape::ToString() const {
  std::stringstream ss;
  ss << '[';
  size_t ndim = dim.size();
  if (ndim > 0) {
    ss << dim[0];
  }
  for (size_t i = 1; i < ndim; i++) {
    ss << ", " << dim[i];
  }
  ss << ']';
  return ss.str();
}

dim_t* Shape::DataPtr() { return dim.data(); }
const dim_t* Shape::DataPtr() const { return dim.data(); }
dim_t& Shape::operator[](int index) { return dim[index]; }
dim_t Shape::operator[](int index) const { return dim[index]; }
void Shape::operator=(const Shape& shape) { dim = shape.dim; }
bool Shape::operator==(const Shape& shape) const { return dim == shape.dim; }
bool Shape::operator!=(const Shape& shape) const { return dim != shape.dim; }

}  // namespace allspark
