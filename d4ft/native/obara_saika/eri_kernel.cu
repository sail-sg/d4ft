#include "eri.h"
#include "eri_kernel.h"
#include "hemi/hemi.h"
#include "hemi/parallel_for.h"
#include <iostream>

HEMI_DEV_CALLABLE int num_unique_ij(int n) { return n * (n + 1) / 2; }
HEMI_DEV_CALLABLE int num_unique_ijkl(int n) {
  return num_unique_ij(num_unique_ij(n));
}
HEMI_DEV_CALLABLE void triu_ij_from_index(int n, int index, int *i,
                                          int *j) {
  int a = 1;
  int b = (2 * n + 1);
  int c = 2 * index;
  int i_ = (int)((b - std::sqrt(b * b - 4 * a * c)) / (2 * a));
  int j_ = int(index - (2 * n + 1 - i_) * i_ / 2 + i_);
  *i = i_;
  *j = j_;
}

HEMI_DEV_CALLABLE void get_symmetry_count(int i, int j, int k, int l, int *count) {
  int count_ = 1;
  if(i == k & j == l){
    if(i != j){
      count_ *= 4;
    }
  } else{
    count_ *= 2;
    if(i == j){
      if(k != l){
        count_ *= 2;
      }
    } else{
      count_ *= 2;
      if(k != l){
        count_ *= 2;
      }
    }
  }
  *count = count_;
}

// template <typename FLOAT>
void Hartree_32::Gpu(cudaStream_t stream, 
                  Array<const int>& N, 
                  Array<const int>& index_4c, 
                  Array<const int>& n, 
                  Array<const float>& r, 
                  Array<const float>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<const int>& ab_range,
                  Array<float>& output) {
  std::cout<<index_4c.spec->shape[0]<<std::endl;
  hemi::ExecutionPolicy ep;
  ep.setStream(stream);
  hemi::parallel_for(ep, 0, index_4c.spec->shape[0], [=] HEMI_LAMBDA(int index) {
    int i, j, k, l, ij, kl;
    // triu_ij_from_index(num_unique_ij(N.ptr[0]), index_4c.ptr[index], &ij, &kl);
    // triu_ij_from_index(N.ptr[0], ij, &i, &j);
    // triu_ij_from_index(N.ptr[0], kl, &k, &l);
    // output.ptr[index] = index_4c.ptr[index];
    i = index_4c.ptr[4*index + 0];
    j = index_4c.ptr[4*index + 1];
    k = index_4c.ptr[4*index + 2];
    l = index_4c.ptr[4*index + 3];
    output.ptr[index] = eri<float>(n.ptr[0 * N.ptr[0] + i], n.ptr[1 * N.ptr[0] + i], n.ptr[2 * N.ptr[0] + i], // a
                           n.ptr[0 * N.ptr[0] + j], n.ptr[1 * N.ptr[0] + j], n.ptr[2 * N.ptr[0] + j], // b
                           n.ptr[0 * N.ptr[0] + k], n.ptr[1 * N.ptr[0] + k], n.ptr[2 * N.ptr[0] + k], // c
                           n.ptr[0 * N.ptr[0] + l], n.ptr[1 * N.ptr[0] + l], n.ptr[2 * N.ptr[0] + l], // d
                           r.ptr[0 * N.ptr[0] + i], r.ptr[1 * N.ptr[0] + i], r.ptr[2 * N.ptr[0] + i], // a
                           r.ptr[0 * N.ptr[0] + j], r.ptr[1 * N.ptr[0] + j], r.ptr[2 * N.ptr[0] + j], // b
                           r.ptr[0 * N.ptr[0] + k], r.ptr[1 * N.ptr[0] + k], r.ptr[2 * N.ptr[0] + k], // c
                           r.ptr[0 * N.ptr[0] + l], r.ptr[1 * N.ptr[0] + l], r.ptr[2 * N.ptr[0] + l], // d
                           z.ptr[i], z.ptr[j], z.ptr[k], z.ptr[l],                   // z
                           min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
  });
}

void Hartree_64::Gpu(cudaStream_t stream, 
                  Array<const int>& N,
                  Array<const int>& thread_load, 
                  Array<const int64_t>& thread_num,
                  Array<const int64_t>& screened_length, 
                  Array<const int>& n, 
                  Array<const double>& r, 
                  Array<const double>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<const int>& sorted_ab_idx,
                  Array<const int>& sorted_cd_idx,
                  Array<const int>& screened_cd_idx_start,
                  //Array<const int>& screened_idx_offset,
                  Array<const int>& ab_thread_num,
                  Array<const int>& ab_thread_offset,
                  Array<const double>& pgto_coeff,
                  Array<const double>& pgto_normalization_factor,
                  Array<const int>& pgto_idx_to_cgto_idx,
                  Array<const double>& rdm1,
                  Array<const int>& n_cgto,
                  Array<const int>& n_pgto,
                  Array<double>& output) {
  int* thread_ab_index;
  int64_t thread_length;
  cudaMemcpy(&thread_length, thread_num.ptr, sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMalloc((void **)&thread_ab_index, thread_length * sizeof(int));
  std::cout<<thread_length<<std::endl;
  int num_cd = sorted_cd_idx.spec->shape[0];
  int* ncd;
  double* rcd;
  double* zcd;
  cudaMalloc((void **)&ncd, 3 * 2 * num_cd * sizeof(int));
  cudaMalloc((void **)&rcd, 3 * 2 * num_cd * sizeof(double));
  cudaMalloc((void **)&zcd, 2 * num_cd * sizeof(double));

  // Pre-screen, result is (ab_index, cd_index), i.e. (ab, cd)
  hemi::ExecutionPolicy ep;
  ep.setStream(stream);
  hemi::parallel_for(ep, 0, ab_thread_num.spec->shape[0], [=] HEMI_LAMBDA(int index) {
    for(int i = 0; i < ab_thread_num.ptr[index]; i++ ){
      int loc;
      loc = ab_thread_offset.ptr[index] + i;
      thread_ab_index[loc] = index;
    }
    __syncthreads();
  });

  // get ncd, rcd, zcd in cd order
  hemi::parallel_for(ep, 0, sorted_cd_idx.spec->shape[0], [=] HEMI_LAMBDA(int index) {
    int cd;
    int c,d;
    cd = sorted_cd_idx.ptr[index];
    triu_ij_from_index(N.ptr[0], cd, &c, &d);
    ncd[0*num_cd + 2 * index] = n.ptr[0 * N.ptr[0] + c];
    ncd[2*num_cd + 2 * index] = n.ptr[1 * N.ptr[0] + c];
    ncd[4*num_cd + 2 * index] = n.ptr[2 * N.ptr[0] + c];
    ncd[0*num_cd + 2 * index + 1] = n.ptr[0 * N.ptr[0] + d];
    ncd[2*num_cd + 2 * index + 1] = n.ptr[1 * N.ptr[0] + d];
    ncd[4*num_cd + 2 * index + 1] = n.ptr[2 * N.ptr[0] + d];
    rcd[0*num_cd + 2 * index] = r.ptr[0 * N.ptr[0] + c];
    rcd[2*num_cd + 2 * index] = r.ptr[1 * N.ptr[0] + c];
    rcd[4*num_cd + 2 * index] = r.ptr[2 * N.ptr[0] + c];
    rcd[0*num_cd + 2 * index + 1] = r.ptr[0 * N.ptr[0] + d];
    rcd[2*num_cd + 2 * index + 1] = r.ptr[1 * N.ptr[0] + d];
    rcd[4*num_cd + 2 * index + 1] = r.ptr[2 * N.ptr[0] + d];
    zcd[0*num_cd + index] = z.ptr[c];
    zcd[1*num_cd + index] = z.ptr[d];

    __syncthreads();
  });

  cudaMemset(output.ptr, 0, sizeof(double));
  // Now we have ab cd, we can compute eri and contract it to output
  // For contract, we need 1. count 2. pgto normalization coeff 3. pgto coeff 4.rdm1 (Mocoeff)
  hemi::parallel_for(ep, 0, thread_length, [=] HEMI_LAMBDA(int index) {
    int a, b, c, d; // pgto 4c idx
    int i, j, k, l; // cgto 4c idx
    int ab_index, cd_index;
    int ab, cd;
    double eri_result;
    double Na, Nb, Nc, Nd;
    double Ca, Cb, Cc, Cd;
    double Mab, Mcd;
    int count;
    int nax, nay, naz, nbx, nby, nbz, ncx, ncy, ncz, ndx, ndy, ndz;
    double rax, ray, raz, rbx, rby, rbz, rcx, rcy, rcz, rdx, rdy, rdz;
    double za, zb, zc, zd;

    ab_index = thread_ab_index[index];
    ab = sorted_ab_idx.ptr[ab_index];
    triu_ij_from_index(N.ptr[0], ab, &a, &b);
    nax = n.ptr[0 * N.ptr[0] + a];
    nay = n.ptr[1 * N.ptr[0] + a];
    naz = n.ptr[2 * N.ptr[0] + a];
    nbx = n.ptr[0 * N.ptr[0] + b];
    nby = n.ptr[1 * N.ptr[0] + b];
    nbz = n.ptr[2 * N.ptr[0] + b];
    rax = r.ptr[0 * N.ptr[0] + a];
    ray = r.ptr[1 * N.ptr[0] + a];
    raz = r.ptr[2 * N.ptr[0] + a];
    rbx = r.ptr[0 * N.ptr[0] + b];
    rby = r.ptr[1 * N.ptr[0] + b];
    rbz = r.ptr[2 * N.ptr[0] + b];
    za = z.ptr[a];
    zb = z.ptr[b];
    Ca = pgto_coeff.ptr[a]; 
    Cb = pgto_coeff.ptr[b];
    Na = pgto_normalization_factor.ptr[a];
    Nb = pgto_normalization_factor.ptr[b];
    i = pgto_idx_to_cgto_idx.ptr[a];
    j = pgto_idx_to_cgto_idx.ptr[b];
    Mab = rdm1.ptr[i*n_cgto.ptr[0] + j];
    eri_result = 0;
    for(int cur_ptr = 0; cur_ptr < thread_load.ptr[0]; cur_ptr++ ){
      cd_index = screened_cd_idx_start.ptr[ab_index] + index % ab_thread_num.ptr[ab_index] + cur_ptr * ab_thread_num.ptr[ab_index];
      if(cd_index < num_cd){
        ncx = ncd[0*num_cd + 2 * cd_index];
        ncy = ncd[2*num_cd + 2 * cd_index];
        ncz = ncd[4*num_cd + 2 * cd_index];
        ndx = ncd[0*num_cd + 2 * cd_index + 1];
        ndy = ncd[2*num_cd + 2 * cd_index + 1];
        ndz = ncd[4*num_cd + 2 * cd_index + 1];
        rcx = rcd[0*num_cd + 2 * cd_index];
        rcy = rcd[2*num_cd + 2 * cd_index];
        rcz = rcd[4*num_cd + 2 * cd_index];
        rdx = rcd[0*num_cd + 2 * cd_index + 1];
        rdy = rcd[2*num_cd + 2 * cd_index + 1];
        rdz = rcd[4*num_cd + 2 * cd_index + 1];
        zc = zcd[0*num_cd + cd_index];
        zd = zcd[1*num_cd + cd_index];

        cd = sorted_cd_idx.ptr[cd_index];
        triu_ij_from_index(N.ptr[0], cd, &c, &d);
        get_symmetry_count(a, b, c, d, &count);
        double dcount = static_cast<double>(count);
        Cc = pgto_coeff.ptr[c];
        Cd = pgto_coeff.ptr[d];
        Nc = pgto_normalization_factor.ptr[c];
        Nd = pgto_normalization_factor.ptr[d];
        k = pgto_idx_to_cgto_idx.ptr[c];
        l = pgto_idx_to_cgto_idx.ptr[d];
        Mcd = rdm1.ptr[k*n_cgto.ptr[0] + l];
        eri_result += eri<double>(nax, nay, naz, // a
                                  nbx, nby, nbz, // b
                                  ncx, ncy, ncz, // c
                                  ndx, ndy, ndz, // d
                                  rax, ray, raz, // a
                                  rbx, rby, rbz, // b
                                  rcx, rcy, rcz, // c
                                  rdx, rdy, rdz, // d
                                  za, zb, zc, zd, // z
                                  min_a.ptr, min_c.ptr,
                                  max_ab.ptr, max_cd.ptr, Ms.ptr) * dcount * Na * Nb * Nc * Nd * Ca * Cb * Cc * Cd * Mab * Mcd;
        // eri_result += dcount * Na * Nb * Nc * Nd * Ca * Cb * Cc * Cd * Mab * Mcd;
      }
    }
    atomicAdd(output.ptr, eri_result);
    __syncthreads();
  });


  
  // Build abcd index list version
/*
  int* ab_cd_idx;
  int64_t ab_cd_idx_length;
  cudaMemcpy(&ab_cd_idx_length, screened_length.ptr, sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMalloc((void **)&ab_cd_idx, 2 * ab_cd_idx_length * sizeof(int));
  std::cout<<ab_cd_idx_length<<std::endl;
  int num_cd = sorted_cd_idx.spec->shape[0];

  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start);

  // Pre-screen, result is (ab_index, cd_index), i.e. (ab, cd)
  hemi::ExecutionPolicy ep;
  ep.setStream(stream);
  hemi::parallel_for(ep, 0, screened_cd_idx_start.spec->shape[0], [=] HEMI_LAMBDA(int index) {
    for(int i = screened_cd_idx_start.ptr[index]; i < num_cd; i++ ){
      int loc;
      loc = screened_idx_offset.ptr[index] + i - screened_cd_idx_start.ptr[index];
      ab_cd_idx[loc] = sorted_ab_idx.ptr[index]; // ab
      ab_cd_idx[loc + screened_length.ptr[0]] = sorted_cd_idx.ptr[i]; // cd
      // output.ptr[loc] = sorted_ab_idx.ptr[index]; // ab
      // output.ptr[loc + screened_length.ptr[0]] = sorted_cd_idx.ptr[i]; // cd
    }
    __syncthreads();
  });

  
  cudaMemset(output.ptr, 0, sizeof(double));
  // Now we have ab cd, we can compute eri and contract it to output
  // For contract, we need 1. count 2. pgto normalization coeff 3. pgto coeff 4.rdm1 (Mocoeff)
  hemi::parallel_for(ep, 0, ab_cd_idx_length, [=] HEMI_LAMBDA(int64_t index) {
    int a, b, c, d; // pgto 4c idx
    int i, j, k, l; // cgto 4c idx
    double eri_result;
    double Na, Nb, Nc, Nd;
    double Ca, Cb, Cc, Cd;
    double Mab, Mcd;
    int count;
    triu_ij_from_index(N.ptr[0], ab_cd_idx[index], &a, &b);
    triu_ij_from_index(N.ptr[0], ab_cd_idx[index + screened_length.ptr[0]], &c, &d);
    get_symmetry_count(a, b, c, d, &count);
    double dcount = static_cast<double>(count);
    // pgto coeff
    Ca = pgto_coeff.ptr[a]; 
    Cb = pgto_coeff.ptr[b];
    Cc = pgto_coeff.ptr[c];
    Cd = pgto_coeff.ptr[d];
    // pgto normalization factor
    Na = pgto_normalization_factor.ptr[a];
    Nb = pgto_normalization_factor.ptr[b];
    Nc = pgto_normalization_factor.ptr[c];
    Nd = pgto_normalization_factor.ptr[d];
    // cgto i j k l
    i = pgto_idx_to_cgto_idx.ptr[a];
    j = pgto_idx_to_cgto_idx.ptr[b];
    k = pgto_idx_to_cgto_idx.ptr[c];
    l = pgto_idx_to_cgto_idx.ptr[d];
    // rdm1_ab, rdm1_cd
    Mab = rdm1.ptr[i*n_cgto.ptr[0] + j];
    Mcd = rdm1.ptr[k*n_cgto.ptr[0] + l];
    eri_result = eri<double>(n.ptr[0 * N.ptr[0] + a], n.ptr[1 * N.ptr[0] + a], n.ptr[2 * N.ptr[0] + a], // a
                           n.ptr[0 * N.ptr[0] + b], n.ptr[1 * N.ptr[0] + b], n.ptr[2 * N.ptr[0] + b], // b
                           n.ptr[0 * N.ptr[0] + c], n.ptr[1 * N.ptr[0] + c], n.ptr[2 * N.ptr[0] + c], // c
                           n.ptr[0 * N.ptr[0] + d], n.ptr[1 * N.ptr[0] + d], n.ptr[2 * N.ptr[0] + d], // d
                           r.ptr[0 * N.ptr[0] + a], r.ptr[1 * N.ptr[0] + a], r.ptr[2 * N.ptr[0] + a], // a
                           r.ptr[0 * N.ptr[0] + b], r.ptr[1 * N.ptr[0] + b], r.ptr[2 * N.ptr[0] + b], // b
                           r.ptr[0 * N.ptr[0] + c], r.ptr[1 * N.ptr[0] + c], r.ptr[2 * N.ptr[0] + c], // c
                           r.ptr[0 * N.ptr[0] + d], r.ptr[1 * N.ptr[0] + d], r.ptr[2 * N.ptr[0] + d], // d
                           z.ptr[a], z.ptr[b], z.ptr[c], z.ptr[d],                   // z
                           min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
    eri_result = eri_result * dcount * Na * Nb * Nc * Nd * Ca * Cb * Cc * Cd * Mab * Mcd;
    // eri_result = Mab;
    // output.ptr[index] = eri_result;
    // eri_result = eri_result * dcount * Na * Nb * Nc * Nd * Ca * Cb * Cc * Cd; // * Mab * Mcd;
    // prod result from rdm1
    atomicAdd(output.ptr, eri_result);
    __syncthreads();
  });
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
  
  // std::cout<<index_4c.spec->shape[0]<<std::endl;
  // hemi::ExecutionPolicy ep;
  // ep.setStream(stream);
  // hemi::parallel_for(ep, 0, index_4c.spec->shape[0], [=] HEMI_LAMBDA(int index) {
  //   int i, j, k, l, ij, kl;
  //   // triu_ij_from_index(num_unique_ij(N.ptr[0]), index_4c.ptr[index], &ij, &kl);
  //   // triu_ij_from_index(N.ptr[0], ij, &i, &j);
  //   // triu_ij_from_index(N.ptr[0], kl, &k, &l);
  //   // output.ptr[index] = index_4c.ptr[index];
  //   i = index_4c.ptr[4*index + 0];
  //   j = index_4c.ptr[4*index + 1];
  //   k = index_4c.ptr[4*index + 2];
  //   l = index_4c.ptr[4*index + 3];
  //   output.ptr[index] = eri<double>(n.ptr[0 * N.ptr[0] + i], n.ptr[1 * N.ptr[0] + i], n.ptr[2 * N.ptr[0] + i], // a
  //                          n.ptr[0 * N.ptr[0] + j], n.ptr[1 * N.ptr[0] + j], n.ptr[2 * N.ptr[0] + j], // b
  //                          n.ptr[0 * N.ptr[0] + k], n.ptr[1 * N.ptr[0] + k], n.ptr[2 * N.ptr[0] + k], // c
  //                          n.ptr[0 * N.ptr[0] + l], n.ptr[1 * N.ptr[0] + l], n.ptr[2 * N.ptr[0] + l], // d
  //                          r.ptr[0 * N.ptr[0] + i], r.ptr[1 * N.ptr[0] + i], r.ptr[2 * N.ptr[0] + i], // a
  //                          r.ptr[0 * N.ptr[0] + j], r.ptr[1 * N.ptr[0] + j], r.ptr[2 * N.ptr[0] + j], // b
  //                          r.ptr[0 * N.ptr[0] + k], r.ptr[1 * N.ptr[0] + k], r.ptr[2 * N.ptr[0] + k], // c
  //                          r.ptr[0 * N.ptr[0] + l], r.ptr[1 * N.ptr[0] + l], r.ptr[2 * N.ptr[0] + l], // d
  //                          z.ptr[i], z.ptr[j], z.ptr[k], z.ptr[l],                   // z
  //                          min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
  // });*/
}

// template <typename FLOAT>
void Hartree_32_uncontracted::Gpu(cudaStream_t stream, 
                  Array<const int>& N, 
                  Array<const int>& index_4c, 
                  Array<const int>& n, 
                  Array<const float>& r, 
                  Array<const float>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<float>& output) {
  // std::cout<<index_4c.spec->shape[0]<<std::endl;
  hemi::ExecutionPolicy ep;
  ep.setStream(stream);
  hemi::parallel_for(ep, 0, index_4c.spec->shape[0], [=] HEMI_LAMBDA(int index) {
    int i, j, k, l, ij, kl;
    // triu_ij_from_index(num_unique_ij(N.ptr[0]), index_4c.ptr[index], &ij, &kl);
    // triu_ij_from_index(N.ptr[0], ij, &i, &j);
    // triu_ij_from_index(N.ptr[0], kl, &k, &l);
    // output.ptr[index] = index_4c.ptr[index];
    i = index_4c.ptr[4*index + 0];
    j = index_4c.ptr[4*index + 1];
    k = index_4c.ptr[4*index + 2];
    l = index_4c.ptr[4*index + 3];
    output.ptr[index] = eri<float>(n.ptr[0 * N.ptr[0] + i], n.ptr[1 * N.ptr[0] + i], n.ptr[2 * N.ptr[0] + i], // a
                           n.ptr[0 * N.ptr[0] + j], n.ptr[1 * N.ptr[0] + j], n.ptr[2 * N.ptr[0] + j], // b
                           n.ptr[0 * N.ptr[0] + k], n.ptr[1 * N.ptr[0] + k], n.ptr[2 * N.ptr[0] + k], // c
                           n.ptr[0 * N.ptr[0] + l], n.ptr[1 * N.ptr[0] + l], n.ptr[2 * N.ptr[0] + l], // d
                           r.ptr[0 * N.ptr[0] + i], r.ptr[1 * N.ptr[0] + i], r.ptr[2 * N.ptr[0] + i], // a
                           r.ptr[0 * N.ptr[0] + j], r.ptr[1 * N.ptr[0] + j], r.ptr[2 * N.ptr[0] + j], // b
                           r.ptr[0 * N.ptr[0] + k], r.ptr[1 * N.ptr[0] + k], r.ptr[2 * N.ptr[0] + k], // c
                           r.ptr[0 * N.ptr[0] + l], r.ptr[1 * N.ptr[0] + l], r.ptr[2 * N.ptr[0] + l], // d
                           z.ptr[i], z.ptr[j], z.ptr[k], z.ptr[l],                   // z
                           min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
  });
}

void Hartree_64_uncontracted::Gpu(cudaStream_t stream, 
                  Array<const int>& N,
                  Array<const int>& index_4c,  
                  Array<const int>& n, 
                  Array<const double>& r, 
                  Array<const double>& z,
                  Array<const int>& min_a, 
                  Array<const int>& min_c, 
                  Array<const int>& max_ab,
                  Array<const int>& max_cd, 
                  Array<const int>& Ms, 
                  Array<double>& output) {
  // std::cout<<index_4c.spec->shape[0]<<std::endl;
  hemi::ExecutionPolicy ep;
  ep.setStream(stream);
  hemi::parallel_for(ep, 0, index_4c.spec->shape[0], [=] HEMI_LAMBDA(int64_t index) {
    int i, j, k, l, ij, kl;
    // triu_ij_from_index(num_unique_ij(N.ptr[0]), index, &ij, &kl);
    // triu_ij_from_index(N.ptr[0], ij, &i, &j);
    // triu_ij_from_index(N.ptr[0], kl, &k, &l);
    // output.ptr[index] = index_4c.ptr[index];
    i = index_4c.ptr[4*index + 0];
    j = index_4c.ptr[4*index + 1];
    k = index_4c.ptr[4*index + 2];
    l = index_4c.ptr[4*index + 3];
    output.ptr[index] = eri<double>(n.ptr[0 * N.ptr[0] + i], n.ptr[1 * N.ptr[0] + i], n.ptr[2 * N.ptr[0] + i], // a
                           n.ptr[0 * N.ptr[0] + j], n.ptr[1 * N.ptr[0] + j], n.ptr[2 * N.ptr[0] + j], // b
                           n.ptr[0 * N.ptr[0] + k], n.ptr[1 * N.ptr[0] + k], n.ptr[2 * N.ptr[0] + k], // c
                           n.ptr[0 * N.ptr[0] + l], n.ptr[1 * N.ptr[0] + l], n.ptr[2 * N.ptr[0] + l], // d
                           r.ptr[0 * N.ptr[0] + i], r.ptr[1 * N.ptr[0] + i], r.ptr[2 * N.ptr[0] + i], // a
                           r.ptr[0 * N.ptr[0] + j], r.ptr[1 * N.ptr[0] + j], r.ptr[2 * N.ptr[0] + j], // b
                           r.ptr[0 * N.ptr[0] + k], r.ptr[1 * N.ptr[0] + k], r.ptr[2 * N.ptr[0] + k], // c
                           r.ptr[0 * N.ptr[0] + l], r.ptr[1 * N.ptr[0] + l], r.ptr[2 * N.ptr[0] + l], // d
                           z.ptr[i], z.ptr[j], z.ptr[k], z.ptr[l],                   // z
                           min_a.ptr, min_c.ptr, max_ab.ptr, max_cd.ptr, Ms.ptr);
  });
}

// HEMI_DEV_CALLABLE size_t num_unique_ij(size_t n) { return n * (n + 1) / 2; }
// HEMI_DEV_CALLABLE size_t num_unique_ijkl(size_t n) {
//   return num_unique_ij(num_unique_ij(n));
// }
// HEMI_DEV_CALLABLE void triu_ij_from_index(size_t n, size_t index, size_t *i,
//                                           size_t *j) {
//   size_t a = 1;
//   size_t b = (2 * n + 1);
//   size_t c = 2 * index;
//   size_t i_ = (size_t)((b - std::sqrt(b * b - 4 * a * c)) / (2 * a));
//   size_t j_ = size_t(index - (2 * n + 1 - i_) * i_ / 2 + i_);
//   *i = i_;
//   *j = j_;
// }

// template <typename FLOAT>
// void hartree(const size_t N, const size_t *n, const FLOAT *r, const FLOAT *z,
//              const size_t *min_a, const size_t *min_c, const size_t *max_ab,
//              const size_t *max_cd, const size_t *Ms, FLOAT *output) {
//   std::cout << num_unique_ijkl(N) << std::endl;
//   hemi::parallel_for(0, num_unique_ijkl(N), [=] HEMI_LAMBDA(int index) {
//     size_t i, j, k, l, ij, kl;
//     triu_ij_from_index(num_unique_ij(N), index, &ij, &kl);
//     triu_ij_from_index(N, ij, &i, &j);
//     triu_ij_from_index(N, kl, &k, &l);
//     float out = eri<float>(n[0 * N + i], n[1 * N + i], n[2 * N + i], // a
//                            n[0 * N + j], n[1 * N + j], n[2 * N + j], // b
//                            n[0 * N + k], n[1 * N + k], n[2 * N + k], // c
//                            n[0 * N + l], n[1 * N + l], n[2 * N + l], // d
//                            r[0 * N + i], r[1 * N + i], r[2 * N + i], // a
//                            r[0 * N + j], r[1 * N + j], r[2 * N + j], // b
//                            r[0 * N + k], r[1 * N + k], r[2 * N + k], // c
//                            r[0 * N + l], r[1 * N + l], r[2 * N + l], // d
//                            z[i], z[j], z[k], z[l],                   // z
//                            min_a, min_c, max_ab, max_cd, Ms);
//   });
// }

// template void hartree<float>(const size_t N, const size_t *n, const float *r,
//                              const float *z, const size_t *min_a,
//                              const size_t *min_c, const size_t *max_ab,
//                              const size_t *max_cd, const size_t *Ms,
//                              float *output);
