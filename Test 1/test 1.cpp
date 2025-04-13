#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <cstring>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

// Your SSE-optimized Boyer-Moore implementation
std::vector<int> boyer_moore_sse_32bytes(const std::string& text, const std::string& pattern) {
    std::vector<int> matches;
    const char* t = text.data();
    const char* p = pattern.data();
    const int n = text.size();
    const int m = pattern.size();

    if (m == 0 || n < m) return matches;

    // Bad character table
    int bad_char[256];
    std::fill_n(bad_char, 256, -1);
    for (int i = 0; i < m; i++) {
        bad_char[(unsigned char)p[i]] = i;
    }

    // Good suffix table
    std::vector<int> good_suffix(m + 1, m);
    std::vector<int> border_pos(m + 1, 0);
    int i = m, j = m + 1;
    border_pos[i] = j;
    
    while (i > 0) {
        while (j <= m && p[i - 1] != p[j - 1]) {
            if (good_suffix[j] == m) {
                good_suffix[j] = j - i;
            }
            j = border_pos[j];
        }
        i--; j--;
        border_pos[i] = j;
    }

    j = border_pos[0];
    for (i = 0; i <= m; i++) {
        if (good_suffix[i] == m) {
            good_suffix[i] = j;
        }
        if (i == j) {
            j = border_pos[j];
        }
    }

    // Special handling for 32-byte patterns
    const bool is_32byte = (m == 32);
    __m128i p_lo, p_hi;
    
    if (is_32byte) {
        p_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        p_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
    }

    int shift = 0;
    while (shift <= n - m) {
        if (is_32byte && (shift <= n - 32)) {
            // Fast path for 32-byte patterns
            __m128i t_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(t + shift));
            __m128i t_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(t + shift + 16));

            unsigned mask_lo = _mm_movemask_epi8(_mm_cmpeq_epi8(p_lo, t_lo));
            unsigned mask_hi = _mm_movemask_epi8(_mm_cmpeq_epi8(p_hi, t_hi));

            if ((mask_lo & mask_hi) == 0xFFFF) {
                matches.push_back(shift);
                shift += good_suffix[0];
                continue;
            } else {
                // Find first mismatch position
                int mismatch_pos;
                if (mask_lo != 0xFFFF) {
                    mismatch_pos = __builtin_ctz(~mask_lo);
                } else {
                    mismatch_pos = 16 + __builtin_ctz(~mask_hi);
                }
                
                int bc_shift = mismatch_pos - bad_char[(unsigned char)t[shift + mismatch_pos]];
                int gs_shift = good_suffix[mismatch_pos + 1];
                shift += std::max(1, std::max(bc_shift, gs_shift));
                continue;
            }
        }

        // Fallback to scalar comparison
        int j = m - 1;
        while (j >= 0 && p[j] == t[shift + j]) {
            j--;
        }

        if (j < 0) {
            matches.push_back(shift);
            shift += good_suffix[0];
        } else {
            int bc_shift = j - bad_char[(unsigned char)t[shift + j]];
            int gs_shift = good_suffix[j + 1];
            shift += std::max(bc_shift, gs_shift);
        }
    }

    return matches;
}

// Standard Turbo Boyer-Moore implementation
std::vector<int> turbo_boyer_moore(const std::string& text, const std::string& pattern) {
    std::vector<int> matches;
    const char* t = text.data();
    const char* p = pattern.data();
    const int n = text.size();
    const int m = pattern.size();

    if (m == 0 || n < m) return matches;

    // Bad character table
    int bad_char[256];
    std::fill_n(bad_char, 256, -1);
    for (int i = 0; i < m; i++) {
        bad_char[(unsigned char)p[i]] = i;
    }

    // Good suffix table
    std::vector<int> good_suffix(m + 1, m);
    std::vector<int> border_pos(m + 1, 0);
    int i = m, j = m + 1;
    border_pos[i] = j;
    
    while (i > 0) {
        while (j <= m && p[i - 1] != p[j - 1]) {
            if (good_suffix[j] == m) {
                good_suffix[j] = j - i;
            }
            j = border_pos[j];
        }
        i--; j--;
        border_pos[i] = j;
    }

    j = border_pos[0];
    for (i = 0; i <= m; i++) {
        if (good_suffix[i] == m) {
            good_suffix[i] = j;
        }
        if (i == j) {
            j = border_pos[j];
        }
    }

    // Turbo shift addition
    int shift = 0;
    int turbo_shift = 0;
    int old_shift = 0;
    
    while (shift <= n - m) {
        int j = m - 1;
        while (j >= 0 && p[j] == t[shift + j]) {
            j--;
            if (turbo_shift != 0 && j == m - 1 - shift + old_shift) {
                j -= turbo_shift;
            }
        }

        if (j < 0) {
            matches.push_back(shift);
            shift += good_suffix[0];
        } else {
            char bc_char = t[shift + j];
            int bc_shift = j - bad_char[(unsigned char)bc_char];
            int gs_shift = good_suffix[j + 1];
            
            old_shift = shift;
            if (shift + m + 1 < n) {
                turbo_shift = m - good_suffix[0];
            } else {
                turbo_shift = 1;
            }
            
            shift += std::max(turbo_shift, std::max(bc_shift, gs_shift));
        }
    }

    return matches;
}

// Generate random DNA sequence
std::string generate_random_dna(size_t length) {
    static const char dna[] = {'A', 'T', 'C', 'G'};
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_int_distribution<> dis(0, 3);
    
    std::string sequence;
    sequence.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        sequence += dna[dis(gen)];
    }
    return sequence;
}

// Verify both algorithms produce the same results
bool verify_results(const std::vector<int>& res1, const std::vector<int>& res2) {
    if (res1.size() != res2.size()) return false;
    for (size_t i = 0; i < res1.size(); ++i) {
        if (res1[i] != res2[i]) return false;
    }
    return true;
}

// Benchmark function
template<typename Func>
void benchmark(const std::string& name, Func search_func, 
               const std::string& text, const std::string& pattern, int iterations) {
    using namespace std::chrono;
    
    // Warm-up
    auto result = search_func(text, pattern);
    
    // Timing
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        result = search_func(text, pattern);
    }
    auto end = high_resolution_clock::now();
    
    // Calculate metrics
    double total_time = duration_cast<microseconds>(end - start).count() / 1e6;
    double avg_time = total_time / iterations;
    double throughput = (text.size() / (1024.0 * 1024.0)) / avg_time;
    
    // Output results
    std::cout << std::setw(20) << name << " | "
              << std::setw(10) << std::fixed << std::setprecision(2) << avg_time * 1000 << " ms | "
              << std::setw(10) << std::fixed << std::setprecision(0) << throughput << " MB/s | "
              << std::setw(8) << result.size() << " matches\n";
}

int main() {
    const size_t text_size = 10'000'000; // 10 million characters
    const int iterations = 100;
    
    // Generate random DNA text
    std::cout << "Generating random DNA sequence (" << text_size << " characters)...\n";
    std::string text = generate_random_dna(text_size);
    
    // Test different pattern sizes
    const std::vector<int> pattern_sizes = {4, 8, 12, 16, 32, 64};
    
    for (int m : pattern_sizes) {
        // Generate a random pattern of size m
        std::string pattern = generate_random_dna(m);
        
        std::cout << "\nBenchmarking with pattern size " << m << ":\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << std::setw(20) << "Algorithm" << " | "
                  << std::setw(10) << "Time" << " | "
                  << std::setw(10) << "Throughput" << " | "
                  << std::setw(8) << "Matches" << "\n";
        std::cout << "------------------------------------------------------------\n";
        
        // Benchmark both algorithms
        benchmark("SSE Boyer-Moore", boyer_moore_sse_32bytes, text, pattern, iterations);
        benchmark("Turbo Boyer-Moore", turbo_boyer_moore, text, pattern, iterations);
        
        // Verify results match
        auto res_sse = boyer_moore_sse_32bytes(text, pattern);
        auto res_turbo = turbo_boyer_moore(text, pattern);
        
        if (!verify_results(res_sse, res_turbo)) {
            std::cerr << "ERROR: Results don't match between algorithms!\n";
        }
    }
    
    return 0;
}