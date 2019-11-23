static inline void floyd_warshall_simd_inner_inline(uniform int distance[],
                                                    uniform size_t n,
                                                    uniform size_t k,
                                                    size_t i) {
    int d_ik = distance[i * n + k];
    foreach (j = 0 ... n) {
        int d_kj = distance[k * n + j];
        int t = d_ik + d_kj;
        size_t ij = i * n + j;
        int d_ij = distance[ij];
        if (t < d_ij) {
            distance[ij] = t;
        }
    }
}

export void floyd_warshall_simd(uniform int distance[],
                                      uniform size_t n, uniform size_t k,
                                      uniform size_t i) {
    floyd_warshall_simd_inner_inline(distance, n, k, i);
}

export void floyd_warshall_seq(uniform int distance[], uniform size_t n) {
    for (uniform size_t k=0; k<n; ++k) {
        for (size_t i=0; i<n; ++i) {
            floyd_warshall_simd_inner_inline(distance, n, k, i); 
        }
    }
}