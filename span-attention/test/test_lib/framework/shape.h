/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    shape.h
 */

#pragma once
#include <string>
#include <vector>

namespace allspark {

typedef int64_t dim_t;
class Shape final {
 public:
  Shape() = default;
  ~Shape() = default;
  Shape(const Shape& shape) = default;
  Shape(Shape&& shape) = default;
  explicit Shape(const std::vector<dim_t>& shape);
  explicit Shape(const std::initializer_list<dim_t>& shape);
  Shape(int ndim, const dim_t* val);

  int Size() const;
  int64_t Count(int start = 0) const;
  int64_t Count(int start, int end) const;
  void Append(dim_t axis);
  std::string ToString() const;
  dim_t* DataPtr();
  const dim_t* DataPtr() const;

  dim_t& operator[](int index);
  dim_t operator[](int index) const;
  void operator=(const Shape& shape);
  bool operator==(const Shape& shape) const;
  bool operator!=(const Shape& shape) const;

 private:
  std::vector<dim_t> dim;
};

inline std::ostream& operator<<(std::ostream& out, Shape const& data) {
  out << data.ToString();
  return out;
}

}  // namespace allspark
