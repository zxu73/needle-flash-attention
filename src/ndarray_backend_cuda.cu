#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ size_t IndexToStrided(size_t i, const CudaVec& shape, const CudaVec& strides,
                                 size_t offset) {
  /**
   * Map the i-th element of a row-major (compact) traversal of an array with the given shape to
   * its position in memory, offset + sum_d idx[d] * strides[d].  The multi-index idx is recovered
   * by peeling off the last dimension first.
   */
  size_t pos = offset;
  for (int d = (int)shape.size - 1; d >= 0; d--) {
    pos += (i % shape.data[d]) * strides.data[d];
    i /= shape.data[d];
  }
  return pos;
}



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of a array
   *   offset: offset of a array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid < size) out[gid] = a[IndexToStrided(gid, shape, strides, offset)];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}



__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  // Inverse of CompactKernel: the gid-th element of compact a goes to its strided spot in out
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[IndexToStrided(gid, shape, strides, offset)] = a[gid];
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[IndexToStrided(gid, shape, strides, offset)] = val;
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}



void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                               VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  // Calculate the global index of the thread.
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);

  // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a', 
  // and store the result in array 'out'.
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
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
__global__ void EwiseKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], b[gid]);
}

template <typename Op>
__global__ void ScalarKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], val);
}

template <typename Op>
__global__ void UnaryKernel(const scalar_t* a, scalar_t* out, size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid]);
}

// device functors: binary ops are shared by the elementwise and scalar versions
struct MulOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x * y; } };
struct DivOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x / y; } };
struct PowerOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return powf(x, y); } };
struct MaximumOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x > y ? x : y; } };
struct EqOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x == y ? 1.0f : 0.0f; } };
struct GeOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x >= y ? 1.0f : 0.0f; } };
struct LogOp { __device__ scalar_t operator()(scalar_t x) const { return logf(x); } };
struct ExpOp { __device__ scalar_t operator()(scalar_t x) const { return expf(x); } };
struct TanhOp { __device__ scalar_t operator()(scalar_t x) const { return tanhf(x); } };

#define DEFINE_EWISE_SCALAR(Name, Op)                                                       \
  void Ewise##Name(const CudaArray& a, const CudaArray& b, CudaArray* out) {               \
    CudaDims dim = CudaOneDim(out->size);                                                  \
    EwiseKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, Op());         \
  }                                                                                        \
  void Scalar##Name(const CudaArray& a, scalar_t val, CudaArray* out) {                    \
    CudaDims dim = CudaOneDim(out->size);                                                  \
    ScalarKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, Op());          \
  }

#define DEFINE_UNARY(Name, Op)                                                              \
  void Ewise##Name(const CudaArray& a, CudaArray* out) {                                   \
    CudaDims dim = CudaOneDim(out->size);                                                  \
    UnaryKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, Op());                \
  }

DEFINE_EWISE_SCALAR(Mul, MulOp)
DEFINE_EWISE_SCALAR(Div, DivOp)
DEFINE_EWISE_SCALAR(Maximum, MaximumOp)
DEFINE_EWISE_SCALAR(Eq, EqOp)
DEFINE_EWISE_SCALAR(Ge, GeOp)
DEFINE_UNARY(Log, LogOp)
DEFINE_UNARY(Exp, ExpOp)
DEFINE_UNARY(Tanh, TanhOp)
#undef DEFINE_EWISE_SCALAR
#undef DEFINE_UNARY

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, PowerOp());
}
/// END SOLUTION


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


// Shared-memory + register tiling: each block computes an MM_BLOCK x MM_BLOCK tile of out using
// MM_THREADS x MM_THREADS threads, and each thread computes an MM_V x MM_V sub-tile held in
// registers.  The reduction dimension is walked in slices of MM_K, which the block cooperatively
// fetches into shared memory.
#define MM_THREADS 16
#define MM_V 4
#define MM_BLOCK (MM_THREADS * MM_V)
#define MM_K 16

__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M,
                             uint32_t N, uint32_t P) {
  __shared__ scalar_t sA[MM_K][MM_BLOCK];  // sA[k][r] = a[row0 + r][k0 + k]
  __shared__ scalar_t sB[MM_K][MM_BLOCK];  // sB[k][c] = b[k0 + k][col0 + c]

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * MM_THREADS + tx;
  const size_t row0 = (size_t)blockIdx.y * MM_BLOCK;
  const size_t col0 = (size_t)blockIdx.x * MM_BLOCK;

  // thread (ty, tx) owns rows ty + i*MM_THREADS and cols tx + j*MM_THREADS of the block tile, so
  // neighbouring threads touch neighbouring columns (conflict-free shared reads, coalesced writes)
  scalar_t c[MM_V][MM_V] = {0};
  scalar_t ra[MM_V], rb[MM_V];

  for (size_t k0 = 0; k0 < N; k0 += MM_K) {
    // cooperative fetch; out-of-range entries are zero-filled so edge tiles need no special case
    for (int l = tid; l < MM_BLOCK * MM_K; l += MM_THREADS * MM_THREADS) {
      int r = l / MM_K, ka = l % MM_K;
      size_t gr = row0 + r, gka = k0 + ka;
      sA[ka][r] = (gr < M && gka < N) ? a[gr * N + gka] : 0.0f;

      int kb = l / MM_BLOCK, cb = l % MM_BLOCK;
      size_t gkb = k0 + kb, gc = col0 + cb;
      sB[kb][cb] = (gkb < N && gc < P) ? b[gkb * P + gc] : 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < MM_K; k++) {
      for (int i = 0; i < MM_V; i++) ra[i] = sA[k][ty + i * MM_THREADS];
      for (int j = 0; j < MM_V; j++) rb[j] = sB[k][tx + j * MM_THREADS];
      for (int i = 0; i < MM_V; i++)
        for (int j = 0; j < MM_V; j++) c[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }

  for (int i = 0; i < MM_V; i++) {
    size_t r = row0 + ty + i * MM_THREADS;
    if (r >= M) break;
    for (int j = 0; j < MM_V; j++) {
      size_t col = col0 + tx + j * MM_THREADS;
      if (col < P) out[r * P + col] = c[i][j];
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 block(MM_THREADS, MM_THREADS);
  dim3 grid((P + MM_BLOCK - 1) / MM_BLOCK, (M + MM_BLOCK - 1) / MM_BLOCK);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////


template <typename Op>
__global__ void ReduceKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size,
                             Op op) {
  // one thread per output element, folding its contiguous block of reduce_size inputs
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= size) return;
  const scalar_t* block = a + gid * reduce_size;
  scalar_t acc = block[0];
  for (size_t j = 1; j < reduce_size; j++) acc = op(acc, block[j]);
  out[gid] = acc;
}

struct AddOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x + y; } };

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size, MaximumOp());
  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size, AddOp());
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// FlashAttention-style fused attention (forward)
////////////////////////////////////////////////////////////////////////////////

// One thread block handles FA_BR query rows of one (batch, head); each thread owns one query row
// and keeps its scaled q, output accumulator o, and softmax statistics (m, l) in registers.  Key
// and value tiles of FA_BC rows are staged through shared memory and shared by the whole block.
#define FA_BR 64
#define FA_BC 32

template <int MAX_D>
__global__ void FlashAttentionFwdKernel(const scalar_t* q, const scalar_t* k, const scalar_t* v,
                                        scalar_t* out, scalar_t* lse, uint32_t Nq, uint32_t Nk,
                                        uint32_t D, scalar_t scale, bool causal) {
  __shared__ scalar_t Ks[FA_BC][MAX_D];
  __shared__ scalar_t Vs[FA_BC][MAX_D];

  const size_t bh = blockIdx.y;
  const int r = threadIdx.x;
  const int64_t i = (int64_t)blockIdx.x * FA_BR + r;
  const bool valid = i < Nq;
  const int64_t diag = (int64_t)Nk - (int64_t)Nq;  // query i sees keys j <= i + diag (causal)

  const scalar_t* Q = q + bh * Nq * D;
  const scalar_t* K = k + bh * Nk * D;
  const scalar_t* V = v + bh * Nk * D;

  // scale is folded into q once; padding dims (d >= D) are zero so they drop out of every dot
  scalar_t q_reg[MAX_D], o[MAX_D];
#pragma unroll
  for (int d = 0; d < MAX_D; d++) {
    q_reg[d] = (valid && d < D) ? Q[i * D + d] * scale : 0.0f;
    o[d] = 0.0f;
  }
  scalar_t m = -INFINITY, l = 0.0f;

  // keys past j_end are invisible to every row in this block under the causal mask
  int64_t j_end = Nk;
  if (causal) {
    int64_t last_row = min((int64_t)blockIdx.x * FA_BR + FA_BR, (int64_t)Nq) - 1;
    j_end = max((int64_t)0, min((int64_t)Nk, last_row + diag + 1));
  }

  for (int64_t j0 = 0; j0 < j_end; j0 += FA_BC) {
    // cooperatively stage K_j, V_j (zero-padded past Nk and D) in shared memory
    for (int idx = r; idx < FA_BC * MAX_D; idx += FA_BR) {
      const int c = idx / MAX_D, d = idx % MAX_D;
      const int64_t j = j0 + c;
      const bool in = j < Nk && d < D;
      Ks[c][d] = in ? K[j * D + d] : 0.0f;
      Vs[c][d] = in ? V[j * D + d] : 0.0f;
    }
    __syncthreads();

    // s = q K_j^T for this row, masked to -inf where not visible
    scalar_t s[FA_BC];
    scalar_t m_new = m;
#pragma unroll
    for (int c = 0; c < FA_BC; c++) {
      scalar_t dot = 0.0f;
#pragma unroll
      for (int d = 0; d < MAX_D; d++) dot += q_reg[d] * Ks[c][d];
      const int64_t j = j0 + c;
      const bool visible = valid && j < Nk && (!causal || j <= i + diag);
      s[c] = visible ? dot : -INFINITY;
      m_new = fmaxf(m_new, s[c]);
    }

    // online softmax update; skipped while nothing has been visible to this row
    if (m_new != -INFINITY) {
      const scalar_t alpha = expf(m - m_new);
      l *= alpha;
#pragma unroll
      for (int d = 0; d < MAX_D; d++) o[d] *= alpha;
#pragma unroll
      for (int c = 0; c < FA_BC; c++) {
        const scalar_t p = expf(s[c] - m_new);
        l += p;
#pragma unroll
        for (int d = 0; d < MAX_D; d++) o[d] += p * Vs[c][d];
      }
      m = m_new;
    }
    __syncthreads();  // the tile is about to be overwritten
  }

  if (valid) {
    const scalar_t inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    scalar_t* out_row = out + (bh * Nq + i) * D;
#pragma unroll
    for (int d = 0; d < MAX_D; d++)
      if (d < D) out_row[d] = o[d] * inv_l;
    lse[bh * Nq + i] = l > 0.0f ? m + logf(l) : -INFINITY;
  }
}

void FlashAttentionFwd(const CudaArray& q, const CudaArray& k, const CudaArray& v, CudaArray* out,
                       CudaArray* lse, uint32_t BH, uint32_t Nq, uint32_t Nk, uint32_t D,
                       scalar_t scale, bool causal) {
  /**
   * Memory-efficient scaled dot-product attention, out = softmax(scale * Q K^T + mask) V, with an
   * online softmax over key tiles; see the CPU backend for the full description of the arguments.
   * The (Nq x Nk) score matrix never exists in global memory.  Supports head dims up to 128.
   */
  if (BH == 0 || Nq == 0) return;
  dim3 grid((Nq + FA_BR - 1) / FA_BR, BH, 1);
  dim3 block(FA_BR, 1, 1);

#define FA_LAUNCH(MD) \
  FlashAttentionFwdKernel<MD>                                                                  \
      <<<grid, block>>>(q.ptr, k.ptr, v.ptr, out->ptr, lse->ptr, Nq, Nk, D, scale, causal)
  if (D <= 16) FA_LAUNCH(16);
  else if (D <= 32) FA_LAUNCH(32);
  else if (D <= 64) FA_LAUNCH(64);
  else if (D <= 128) FA_LAUNCH(128);
  else throw std::runtime_error("flash_attention_fwd: head dim > 128 is not supported on cuda");
#undef FA_LAUNCH

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  m.def("flash_attention_fwd", FlashAttentionFwd);
}
