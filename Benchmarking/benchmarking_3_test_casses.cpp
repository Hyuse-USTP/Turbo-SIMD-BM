#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <emmintrin.h>
#include <bitset>
#include <fstream>
#include <algorithm>
#include <cmath>

using namespace std;

// Standard Boyer-Moore algorithm with bad character and good suffix heuristics
vector<int> bad_character_heuristic(const string& pattern) {
    vector<int> bad_char(256, -1);
    for (int i = 0; i < pattern.size(); i++) {
        bad_char[(unsigned char)pattern[i]] = i;
    }
    return bad_char;
}

vector<int> compute_good_suffix(const string& pattern) {
    int m = pattern.size();
    vector<int> good_suffix(m + 1, m);
    vector<int> border(m + 1);

    int i = m, j = m + 1;
    border[i] = j;
    while (i > 0) {
        while (j <= m && pattern[i - 1] != pattern[j - 1]) {
            if (good_suffix[j] == m) {
                good_suffix[j] = j - i;
            }
            j = border[j];
        }
        i--; j--;
        border[i] = j;
    }

    j = border[0];
    for (i = 0; i <= m; i++) {
        if (good_suffix[i] == m) {
            good_suffix[i] = j;
        }
        if (i == j) {
            j = border[j];
        }
    }

    return good_suffix;
}

vector<int> boyer_moore(const string& text, const string& pattern) {
    vector<int> bad_char = bad_character_heuristic(pattern);
    vector<int> good_suffix = compute_good_suffix(pattern);
    vector<int> matches;
    int n = text.size(), m = pattern.size();

    int shift = 0;
    while (shift <= n - m) {
        int j = m - 1;
        while (j >= 0 && pattern[j] == text[shift + j]) {
            j--;
        }

        if (j < 0) {
            matches.push_back(shift);
            shift += (good_suffix[0] > 0) ? good_suffix[0] : 1;
        } else {
            int bc_shift = max(1, j - bad_char[text[shift + j]]);
            int gs_shift = good_suffix[j + 1];
            shift += max(bc_shift, gs_shift);
        }
    }
    return matches;
}

// Boyer-Moore-Horspool (BMH) algorithm
vector<int> BMHAlgorithm(const string& text, const string& pattern) {
    vector<int> matches;
    int n = text.size();
    int m = pattern.size();
    if (m == 0 || n < m) return matches;

    int badCharShift[256];
    for (int i = 0; i < 256; i++) badCharShift[i] = m;
    for (int i = 0; i < m - 1; i++) {
        badCharShift[(unsigned char)pattern[i]] = m - 1 - i;
    }

    int i = 0;
    while (i <= n - m) {
        int j = m - 1;
        while (j >= 0 && pattern[j] == text[i + j]) {
            j--;
        }

        if (j < 0) {
            matches.push_back(i);
            i++; 
        } else {
            i += badCharShift[(unsigned char)text[i + m - 1]];
        }
    }
    return matches;
}

// Turbo Boyer-Moore (Turbo BM) algorithm
vector<int> TurboBM(const string& text, const string& pattern) {
    vector<int> matches;
    int n = text.size();
    int m = pattern.size();
    if (m == 0 || n < m) return matches;

    int bad_char[256];
    for (int i = 0; i < 256; i++) bad_char[i] = -1;
    for (int i = 0; i < m; i++) {
        bad_char[(unsigned char)pattern[i]] = i;
    }

    vector<int> good_suffix(m + 1, m);
    vector<int> border_pos(m + 1, 0);

    int i = m, j = m + 1;
    border_pos[i] = j;
    while (i > 0) {
        while (j <= m && pattern[i - 1] != pattern[j - 1]) {
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

    int shift = 0;
    int prev_match_len = 0;  // Track previous match length

    while (shift <= n - m) {
        int curr_match_len = 0;
        int j = m - 1;
        while (j >= 0 && pattern[j] == text[shift + j]) {
            j--;
            curr_match_len++;  // Count matched characters
        }

        if (j < 0) {
            matches.push_back(shift);
            shift += good_suffix[0];
            prev_match_len = 0;  // Reset after full match
        } else {
            int bc_shift = max(1, j - bad_char[(unsigned char)text[shift + j]]);
            int gs_shift = good_suffix[j + 1];
            
            // Turbo shift condition (key addition)
            int turbo_shift = (curr_match_len > prev_match_len) 
                            ? (prev_match_len + 1) 
                            : gs_shift;

            shift += max(bc_shift, turbo_shift);
            prev_match_len = curr_match_len;  // Update for next iteration
        }
    }
    return matches;
}

// Shift-Or implementation with 256-element array
vector<int> ShiftOr(const string& text, const string& pattern) {
    vector<int> matches;
    int m = pattern.length();
    int n = text.length();
    
    if (m == 0) return matches;
    
    unsigned int mask[256];
    for (int i = 0; i < 256; i++) {
        mask[i] = ~0U;
    }
    
    for (int i = 0; i < m; i++) {
        mask[(unsigned char)pattern[i]] &= ~(1U << i);
    }
    
    unsigned int state = ~0U;
    unsigned int goal = 1U << (m - 1);
    
    for (int i = 0; i < n; i++) {
        state = (state << 1) | mask[(unsigned char)text[i]];
        if ((state & goal) == 0) {
            matches.push_back(i - m + 1);
        }
    }
    
    return matches;
}

// Boyer-Moore and SIMD Optimized Shift-Or Hybrid
vector<int> ShiftSIMD(const string& text, const string& pattern) {
    vector<int> matches;
    const char* t = text.data();
    const char* p = pattern.data();
    const int n = text.size();
    const int m = pattern.size();
    if (m == 0) return matches;

    unsigned int mask[256];
    for (int i = 0; i < 256; i++) {
        mask[i] = ~0U;
    }
    
    for (int i = 0; i < m; i++) {
        mask[(unsigned char)p[i]] &= ~(1U << i);
    }

    unsigned int state = ~0U;
    const unsigned int goal = 1U << (m - 1);

    const int simd_width = 16;
    const bool use_simd = m > simd_width;
    __m128i p_vec;
    if (use_simd) {
        p_vec = _mm_loadu_si128((const __m128i*)p);
    }
    for (int i = 0; i < n; i++) {
        if (use_simd && i <= n - 16 && (state & 0x8000) == 0) {
            __m128i t_vec = _mm_loadu_si128((const __m128i*)(t + i));
            unsigned simd_result = _mm_movemask_epi8(_mm_cmpeq_epi8(p_vec, t_vec));
            
            if (simd_result != 0xFFFF) {
                int skip = __builtin_ctz(~simd_result);
                i += skip;
                state = ~0U;
                continue;
            }
            bool match = true;
            for (int k = simd_width; k < m; k++) {
                if (t[i + k] != p[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                matches.push_back(i);
                i += m - 1;
                state = ~0U;
                continue;
            }
        }
        state = (state << 1) | mask[(unsigned char)t[i]];
        if ((state & goal) == 0) {
            matches.push_back(i - m + 1);
        }
    }
    return matches;
}

bool is_extremely_repetitive(const string& pattern) {
    const size_t len = pattern.size();
    if (len <= 4) return false;

    // Check for pure homopolymers (AAAA, CCCCC, etc.)
    const char first = pattern[0];
    bool pure_homopolymer = true;
    for (size_t i = 1; i < len; i++) {
        if (pattern[i] != first) {
            pure_homopolymer = false;
            break;
        }
    }
    if (pure_homopolymer) return true;

    // Check for simple microsatellites (ATATAT, CGGCGGCGG, etc.)
    for (size_t k = 1; k <= 3 && k <= len/2; k++) {
        bool perfect_repeat = true;
        for (size_t i = k; i < len; i++) {
            if (pattern[i] != pattern[i % k]) {
                perfect_repeat = false;
                break;
            }
        }
        if (perfect_repeat) return true;
    }

    return false;
}

vector<int> ShiftBM_Hybrid(const string& text, const string& pattern) {
    vector<int> matches;
    const char* t = text.data();
    const char* p = pattern.data();
    const int n = text.size();
    const int m = pattern.size();
    if (m == 0 || n < m) return matches;

    bool use_shift_or = m <= 16 || is_extremely_repetitive(pattern); 

    if (use_shift_or) {
        return ShiftSIMD(text, pattern);
    }

    int bad_char[256];
    fill_n(bad_char, 256, -1);
    for (int i = 0; i < m; i++) {
        bad_char[(unsigned char)p[i]] = i;
    }

    vector<int> good_suffix(m + 1, m);
    vector<int> border_pos(m + 1, 0);
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

    int shift = 0;
    while (shift <= n - m) {
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
            shift += max(bc_shift, gs_shift);
        }
    }

    return matches;
}

// Knuth-Morris-Pratt (KMP) algorithm
vector<int> computeLPS(const string& pattern) {
    int m = pattern.length();
    vector<int> lps(m, 0);
    int len = 0;
    int i = 1;
    
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    return lps;
}

vector<int> KMP(const string& text, const string& pattern) {
    vector<int> matches;
    int n = text.length();
    int m = pattern.length();
    
    if (m == 0 || n < m) return matches;
    
    vector<int> lps = computeLPS(pattern);
    int i = 0; // index for text
    int j = 0; // index for pattern
    
    while (i < n) {
        if (pattern[j] == text[i]) {
            i++;
            j++;
        }
        
        if (j == m) {
            matches.push_back(i - j);
            j = lps[j - 1];
        } else if (i < n && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i++;
            }
        }
    }
    return matches;
}

// Rabin-Karp algorithm
const int BASE = 256;
const int PRIME = 101;

vector<int> RabinKarp(const string& text, const string& pattern) {
    vector<int> matches;
    int n = text.length();
    int m = pattern.length();
    
    if (m == 0 || n < m) return matches;
    
    int patternHash = 0;
    int textHash = 0;
    int h = 1;
    
    for (int i = 0; i < m - 1; i++) {
        h = (h * BASE) % PRIME;
    }
    
    for (int i = 0; i < m; i++) {
        patternHash = (BASE * patternHash + pattern[i]) % PRIME;
        textHash = (BASE * textHash + text[i]) % PRIME;
    }
    
    for (int i = 0; i <= n - m; i++) {
        if (patternHash == textHash) {
            bool match = true;
            for (int j = 0; j < m; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                matches.push_back(i);
            }
        }
        
        if (i < n - m) {
            textHash = (BASE * (textHash - text[i] * h) + text[i + m]) % PRIME;
            if (textHash < 0) {
                textHash += PRIME;
            }
        }
    }
    return matches;
}

string generate_dna(int length, int mode) {
    const string bases = "ACGT";
    string dna;
    dna.reserve(length);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> base_dis(0, 3);
    uniform_int_distribution<> len_dis(1, 3); // For period lengths 1-3

    // Mode 0: Non-repetitive random DNA (control)
    if (mode == 0) {
        for (int i = 0; i < length; i++) {
            dna += bases[base_dis(gen)];
        }
        return dna;
    }

    // Mode 1: Paper-aligned repetitive DNA (periods 1-3)
    while (dna.size() < length) {
        int period = len_dis(gen); // Randomly choose period 1, 2, or 3
        string pattern;
        
        // Generate repeating unit
        for (int p = 0; p < period; p++) {
            pattern += bases[base_dis(gen)];
        }

        // Insert repeats with realistic lengths
        int repeat_count = 5 + (base_dis(gen) % 15); // 5-20 repeats
        int remaining = length - dna.size();
        int insert_length = min(period * repeat_count, remaining);

        // Alternate between repetitive and short random segments
        if (mode == 1) {
            // Pure period-1-3 repeats (e.g., AAAA, CACACAC, GAAGAAGAA)
            while (insert_length-- > 0) {
                dna += pattern[insert_length % period];
            }
        } else if (mode == 2) {
            // Mixed-mode (realistic): repeats with random spacers
            for (int i = 0; i < insert_length; i++) {
                if (i % (period + 2) == 0) { // Insert random base occasionally
                    dna += bases[base_dis(gen)];
                } else {
                    dna += pattern[i % period];
                }
            }
        }
    }

    return dna.substr(0, length);
}

const string DNA_BASES = "ACGT";
const int DNA_ALPHABET_SIZE = 4;

string generateRandomDNA(size_t length) {
    string dna;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, DNA_ALPHABET_SIZE - 1);
    
    for (size_t i = 0; i < length; i++) {
        dna += DNA_BASES[dis(gen)];
    }
    return dna;
}

void warmup() {
    string warmup_text = generate_dna(10000, 2);  // Mixed-mode DNA
    vector<string> warmup_patterns = {
        "AAAAA",        // Period-1
        "CACACA",       // Period-2 
        "GAAGAA",       // Period-3
        "ACGTACGTACGT"  // Non-repetitive
    };

    for (const auto& pattern : warmup_patterns) {
        boyer_moore(warmup_text, pattern);
        BMHAlgorithm(warmup_text, pattern);
        TurboBM(warmup_text, pattern);
        ShiftOr(warmup_text, pattern);
        ShiftBM_Hybrid(warmup_text, pattern);
        KMP(warmup_text, pattern);
        RabinKarp(warmup_text, pattern);
    }
}

void run_benchmark(const string& genome, const string& name) {
    vector<string> patterns = {
        generateRandomDNA(4),               // 4          
        generateRandomDNA(12),              // 12 Non-repetitive
        generateRandomDNA(16),              // 16 Non-repetitive
        generateRandomDNA(24),              // 24 Non-repetitive
        generateRandomDNA(32),              // 32 Non-repetitive
    };

    cout << "\n=== Benchmarking on " << name << " (" << genome.size() << "bp) ===" << endl;
    cout << "Algorithm               | Length | Matches | Time (ms)" << endl;
    cout << "------------------------|--------|---------|----------" << endl;

    for (const string& pattern : patterns) {
        int pattern_len = pattern.size();
        string pattern_type;

        auto benchmark = [&](const string& algo_name, auto algo_func) {
            auto start = chrono::high_resolution_clock::now();
            vector<int> matches = algo_func(genome, pattern);
            auto end = chrono::high_resolution_clock::now();
            double time = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
            printf("%-23s | %6d | %7d | %8.4f\n", 
                   algo_name.c_str(), pattern_len, (int)matches.size(), time);
        };

        benchmark("Boyer-Moore", boyer_moore);
        benchmark("BM-Horspool", BMHAlgorithm);
        benchmark("TurboBM", TurboBM);
        benchmark("Shift-Or", ShiftOr);
        benchmark("ShiftBM_Hybrid", ShiftBM_Hybrid);
        benchmark("KMP", KMP);
        benchmark("Rabin-Karp", RabinKarp);

        cout << "------------------------|--------|---------|----------" << endl;
    }
}

int main() {
    const int genome_size = 10'000'000;
    
    cout << "Generating benchmark datasets..." << endl;
    string random_dna = generate_dna(genome_size, 0);  // Non-repetitive
    string repetitive_dna = generate_dna(genome_size, 1);  // Pure periods 1-3
    string mixed_dna = generate_dna(genome_size, 2);  // Realistic mix
    
    // Warm up all algorithms
    cout << "Warming up..." << endl;
    warmup();
    
    // Run benchmarks
    run_benchmark(random_dna, "Non-repetitive DNA");
    run_benchmark(repetitive_dna, "Pure Repetitive DNA (P1-P3)");
    run_benchmark(mixed_dna, "Realistic Mixed DNA");
    
    cout << "\nBenchmarking complete!" << endl;
    return 0;
}