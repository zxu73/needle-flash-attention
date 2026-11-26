#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



/**
 * Visit every element of a strided (possibly non-compact) array in row-major order.  Acts like a
 * variable number of nested for loops: idx holds the current multi-index and is incremented with a
 * carry from the last dimension.  For the cnt-th element visited, calls fn(cnt, strided_pos), where
 * strided_pos = offset + sum_i idx[i] * strides[i].
 */
template <typename F>
void ForEachStrided(const std::vector<int32_t>& shape, const std::vector<int32_t>& strides,
                    size_t offset, F fn) {
  size_t ndim = shape.size();
  size_t total = 1;
  for (int32_t d : shape) total *= d;
  if (total == 0) return;

  std::vector<int32_t> idx(ndim, 0);
  size_t pos = offset;
  for (size_t cnt = 0; cnt < total; cnt++) {
    fn(cnt, pos);
    // increment the multi-index, carrying into earlier dimensions
    for (int d = (int)ndim - 1; d >= 0; d--) {
      idx[d]++;
      pos += strides[d];
      if (idx[d] < shape[d]) break;
      pos -= (size_t)idx[d] * strides[d];
      idx[d] = 0;
    }
  }
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  ForEachStrided(shape, strides, offset,
                 [&](size_t cnt, size_t pos) { out->ptr[cnt] = a.ptr[pos]; });
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  ForEachStrided(shape, strides, offset,
                 [&](size_t cnt, size_t pos) { out->ptr[pos] = a.ptr[cnt]; });
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  ForEachStrided(shape, strides, offset, [&](size_t, size_t pos) { out->ptr[pos] = val; });
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN SOLUTION
template <typename Op>
void EwiseApply(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}

template <typename Op>
void ScalarApply(const AlignedArray& a, scalar_t val, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], val);
  }
}

template <typename Op>
void UnaryApply(const AlignedArray& a, AlignedArray* out, Op op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i]);
  }
}

// binary ops shared by the elementwise and scalar versions
inline scalar_t MulOp(scalar_t x, scalar_t y) { return x * y; }
inline scalar_t DivOp(scalar_t x, scalar_t y) { return x / y; }
inline scalar_t PowerOp(scalar_t x, scalar_t y) { return std::pow(x, y); }
inline scalar_t MaximumOp(scalar_t x, scalar_t y) { return x > y ? x : y; }
inline scalar_t EqOp(scalar_t x, scalar_t y) { return x == y ? 1.0f : 0.0f; }
inline scalar_t GeOp(scalar_t x, scalar_t y) { return x >= y ? 1.0f : 0.0f; }

#define DEFINE_EWISE_SCALAR(Name, Op)                                                  \
  void Ewise##Name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
    EwiseApply(a, b, out, Op);                                                        \
  }                                                                                   \
  void Scalar##Name(const AlignedArray& a, scalar_t val, AlignedArray* out) {         \
    ScalarApply(a, val, out, Op);                                                     \
  }

DEFINE_EWISE_SCALAR(Mul, MulOp)
DEFINE_EWISE_SCALAR(Div, DivOp)
DEFINE_EWISE_SCALAR(Maximum, MaximumOp)
DEFINE_EWISE_SCALAR(Eq, EqOp)
DEFINE_EWISE_SCALAR(Ge, GeOp)
#undef DEFINE_EWISE_SCALAR

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarApply(a, val, out, PowerOp);
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  UnaryApply(a, out, [](scalar_t x) { return std::log(x); });
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  UnaryApply(a, out, [](scalar_t x) { return std::exp(x); });
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  UnaryApply(a, out, [](scalar_t x) { return std::tanh(x); });
}
/// END SOLUTION


void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < (size_t)m * p; i++) out->ptr[i] = 0;
  for (size_t i = 0; i < m; i++) {
    for (size_t k = 0; k < n; k++) {
      scalar_t a_ik = a.ptr[i * n + k];
      for (size_t j = 0; j < p; j++) {
        out->ptr[i * p + j] += a_ik * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (size_t i = 0; i < TILE; i++) {
    for (size_t k = 0; k < TILE; k++) {
      float a_ik = a[i * TILE + k];
      for (size_t j = 0; j < TILE; j++) {
        out[i * TILE + j] += a_ik * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  const size_t mt = m / TILE, nt = n / TILE, pt = p / TILE;
  const size_t tile_elems = TILE * TILE;
  for (size_t i = 0; i < (size_t)m * p; i++) out->ptr[i] = 0;
  for (size_t i = 0; i < mt; i++) {
    for (size_t j = 0; j < pt; j++) {
      float* out_tile = out->ptr + (i * pt + j) * tile_elems;
      for (size_t k = 0; k < nt; k++) {
        AlignedDot(a.ptr + (i * nt + k) * tile_elems, b.ptr + (k * pt + j) * tile_elems, out_tile);
      }
    }
  }
  /// END SOLUTION
}

/**
 * Reduce each contiguous block of reduce_size elements of a into one element of out: the block is
 * folded left to right with op, starting from its first element.
 */
template <typename Op>
void ReduceApply(const AlignedArray& a, AlignedArray* out, size_t reduce_size, Op op) {
  for (size_t i = 0; i < out->size; i++) {
    const scalar_t* block = a.ptr + i * reduce_size;
    scalar_t acc = block[0];
    for (size_t j = 1; j < reduce_size; j++) {
      acc = op(acc, block[j]);
    }
    out->ptr[i] = acc;
  }
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  ReduceApply(a, out, reduce_size, MaximumOp);
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  ReduceApply(a, out, reduce_size, [](scalar_t x, scalar_t y) { return x + y; });
  /// END SOLUTION
}


////////////////////////////////////////////////////////////////////////////////
// FlashAttention-style fused attention (forward)
////////////////////////////////////////////////////////////////////////////////

void FlashAttentionFwd(const AlignedArray& q, const AlignedArray& k, const AlignedArray& v,
                       AlignedArray* out, AlignedArray* lse, uint32_t BH, uint32_t Nq, uint32_t Nk,
                       uint32_t D, scalar_t scale, bool causal, uint32_t Br, uint32_t Bc) {
  /**
   * Memory-efficient scaled dot-product attention, out = softmax(scale * Q K^T + mask) V, computed
   * tile by tile with an online softmax so the (Nq x Nk) score matrix is never materialized.
   *
   * Every query tile Q_i (Br rows) keeps a running row max m, running denominator l and an
   * unnormalized output accumulator O.  For each key/value tile (K_j, V_j) of Bc rows:
   *     S     = scale * Q_i K_j^T                (Br x Bc, masked to -inf where not visible)
   *     m_new = max(m, rowmax(S))
   *     P     = exp(S - m_new)
   *     l     = exp(m - m_new) * l + rowsum(P)
   *     O     = exp(m - m_new) * O + P V_j
   * and at the end out = O / l and lse = m + log(l) (kept for the backward pass).
   * Workspace is O(Br * (Bc + D)), independent of the sequence lengths.
   *
   * Args:
   *   q: compact (BH, Nq, D) queries;  k, v: compact (BH, Nk, D) keys / values
   *   out: compact (BH, Nq, D) output;  lse: compact (BH, Nq) row log-sum-exp of the scores
   *   scale: multiplier on Q K^T, usually 1/sqrt(D)
   *   causal: if true, query i only sees keys j <= i + (Nk - Nq) (bottom-right aligned, which is
   *           the convention of nn.MultiHeadAttention's causal mask).  Rows that see no key at all
   *           produce out = 0 and lse = -inf.
   *   Br, Bc: query / key tile sizes
   */
  const scalar_t NEG_INF = -INFINITY;
  const int64_t diag = (int64_t)Nk - (int64_t)Nq;

  std::vector<scalar_t> S((size_t)Br * Bc), O((size_t)Br * D), m(Br), l(Br);

  for (size_t bh = 0; bh < BH; bh++) {
    const scalar_t* Q = q.ptr + bh * Nq * D;
    const scalar_t* K = k.ptr + bh * Nk * D;
    const scalar_t* V = v.ptr + bh * Nk * D;

    for (uint32_t i0 = 0; i0 < Nq; i0 += Br) {
      const uint32_t br = std::min(Br, Nq - i0);
      std::fill(O.begin(), O.end(), 0.0f);
      std::fill(m.begin(), m.end(), NEG_INF);
      std::fill(l.begin(), l.end(), 0.0f);

      // keys past j_end are invisible to every row of this tile under the causal mask
      int64_t j_end = Nk;
      if (causal) j_end = std::max<int64_t>(0, std::min<int64_t>(Nk, (int64_t)i0 + br + diag));

      for (int64_t j0 = 0; j0 < j_end; j0 += Bc) {
        const uint32_t bc = (uint32_t)std::min<int64_t>(Bc, j_end - j0);

        // S = scale * Q_i K_j^T, with masked entries set to -inf
        for (uint32_t r = 0; r < br; r++) {
          const scalar_t* q_row = Q + (size_t)(i0 + r) * D;
          const int64_t last_visible = (int64_t)(i0 + r) + diag;
          for (uint32_t c = 0; c < bc; c++) {
            const int64_t j = j0 + c;
            if (causal && j > last_visible) {
              S[r * Bc + c] = NEG_INF;
              continue;
            }
            const scalar_t* k_row = K + (size_t)j * D;
            scalar_t dot = 0;
            for (uint32_t d = 0; d < D; d++) dot += q_row[d] * k_row[d];
            S[r * Bc + c] = dot * scale;
          }
        }

        // online softmax update of (m, l, O) with this tile
        for (uint32_t r = 0; r < br; r++) {
          scalar_t* s_row = &S[r * Bc];
          scalar_t m_new = m[r];
          for (uint32_t c = 0; c < bc; c++) m_new = std::max(m_new, s_row[c]);
          if (m_new == NEG_INF) continue;  // nothing visible to this row yet

          const scalar_t alpha = std::exp(m[r] - m_new);  // exp(-inf) = 0 on the first hit
          scalar_t row_sum = 0;
          for (uint32_t c = 0; c < bc; c++) {
            s_row[c] = std::exp(s_row[c] - m_new);  // S now holds P
            row_sum += s_row[c];
          }
          l[r] = alpha * l[r] + row_sum;

          scalar_t* o_row = &O[(size_t)r * D];
          for (uint32_t d = 0; d < D; d++) o_row[d] *= alpha;
          for (uint32_t c = 0; c < bc; c++) {
            const scalar_t p = s_row[c];
            const scalar_t* v_row = V + (size_t)(j0 + c) * D;
            for (uint32_t d = 0; d < D; d++) o_row[d] += p * v_row[d];
          }
          m[r] = m_new;
        }
      }

      // normalize and write back this query tile
      for (uint32_t r = 0; r < br; r++) {
        const size_t row = bh * Nq + i0 + r;
        const scalar_t inv_l = l[r] > 0 ? 1.0f / l[r] : 0.0f;
        for (uint32_t d = 0; d < D; d++) out->ptr[row * D + d] = O[(size_t)r * D + d] * inv_l;
        lse->ptr[row] = l[r] > 0 ? m[r] + std::log(l[r]) : NEG_INF;
      }
    }
  }
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  m.def("flash_attention_fwd", FlashAttentionFwd);
}
