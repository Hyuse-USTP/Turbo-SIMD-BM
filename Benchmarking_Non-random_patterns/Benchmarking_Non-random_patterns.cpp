#include <iostream>
#include <vector>  
#include <string>
#include <chrono>
#include <random>
#include <emmintrin.h>
#include <immintrin.h>
#include <bitset>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstring>

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

// Shift-Or for SIMD Boyer-Moore
vector<int> ShiftOrforSIMDBM(const string& text, const string& pattern) {
    vector<int> matches;
    const char* t = text.data();
    const char* p = pattern.data();
    const int n = text.size();
    const int m = pattern.size();
    if (m == 0) return matches;

    unsigned int mask[256];
    for (int i = 0; i < 256; i++) mask[i] = ~0U;
    for (int i = 0; i < m; i++) {
        mask[(unsigned char)pattern[i]] &= ~(1U << i);
    }

    const unsigned int goal = 1U << (m - 1);
    unsigned int state = ~0U;

    for (int i = 0; i < n; i++) {
        state = (state << 1) | mask[(unsigned char)t[i]];
        if ((state & goal) == 0) {
            matches.push_back(i - m + 1);
        }
    }
    return matches;
}

bool is_repetitive(const string& pattern) {
    const size_t len = pattern.size();
    if (len <= 4) return false;

    // 1. Check pure homopolymers (AAAAA, CCCCC, etc.)
    const char first = pattern[0];
    bool pure_homopolymer = true;
    for (size_t i = 1; i < len; i++) {
        if (pattern[i] != first) {
            pure_homopolymer = false;
            break;
        }
    }
    if (pure_homopolymer) return true;

    // 2. Check dinucleotide repeats only (e.g., CACACACA)
    if (len >= 4) {
        const char a = pattern[0];
        const char b = pattern[1];
        bool perfect_dinucleotide = true;
        
        for (size_t i = 2; i < len; i++) {
            char expected = (i % 2 == 0) ? a : b;
            if (pattern[i] != expected) {
                perfect_dinucleotide = false;
                break;
            }
        }
        
        if (perfect_dinucleotide) {
            return true;
        }
    }
    return false;
}

vector<int> SIMDBM(const string& text, const string& pattern) {
    vector<int> matches;
    const char* t = text.data();
    const char* p = pattern.data();
    const int n = text.size();
    const int m = pattern.size();
    if (m == 0 || m > 32) return matches;

    bool use_shift_or = m <= 16;

    if (use_shift_or) {
        return ShiftOrforSIMDBM(text, pattern);
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

    const bool use_simd = is_repetitive(pattern);
    const int simd_width = 16;
    __m128i p_simd1 = _mm_loadu_si128((const __m128i*)p);
    __m128i p_simd2;
    if (m > simd_width) {
        p_simd2 = _mm_loadu_si128((const __m128i*)(p + simd_width));
    }

    i = 0;
    while (i <= n - m) {
        if (use_simd) {
            if (p[m - 1] != t[i + m - 1]) {
                int bc_skip = bad_char[(unsigned char)t[i + m - 1]];
                i += max(1, (bc_skip == -1) ? m : (m - 1 - bc_skip));
                continue;
            }
            
            bool match = true;
            __m128i t_simd1 = _mm_loadu_si128((const __m128i*)(t + i));
            uint32_t mask1 = _mm_movemask_epi8(_mm_cmpeq_epi8(p_simd1, t_simd1));
            uint32_t expected_mask1 = (1U << min(m, simd_width)) - 1;

            if ((mask1 & expected_mask1) != expected_mask1) {
                match = false;
            }
            if (match && m > simd_width) {
                __m128i t_simd2 = _mm_loadu_si128((const __m128i*)(t + i + simd_width));
                uint32_t mask2 = _mm_movemask_epi8(_mm_cmpeq_epi8(p_simd2, t_simd2));
                uint32_t expected_mask2 = (1U << (m - simd_width)) - 1;
                if ((mask2 & expected_mask2) != expected_mask2) {
                    match = false;
                }
            }
            if (match) {
                matches.push_back(i);
                i += good_suffix[0];
            } else {
                int j = m - 1;
                while (j >= 0 && p[j] == t[i + j]) {
                    j--;
                }
                i += max(1, good_suffix[j + 1]);
            }
        } else {
            int j = m - 1;
            while (j >= 0 && p[j] == t[i + j]) {
                j--;
            }
            if (j < 0) {
                matches.push_back(i);
                i += good_suffix[0];
            } else {
                int bc_shift = j - bad_char[(unsigned char)t[i + j]];
                int gs_shift = good_suffix[j + 1];
                i += max(bc_shift, gs_shift);
            }
        }
    }
    return matches;
}


string read_fasta(const string& filepath) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error: Could not open FASTA file." << endl;
        return "";
    }

    string sequence, line;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '>') continue; // Skip header
        sequence += line;
    }
    file.close();
    return sequence;
}


void run_benchmark(const string& genome, const string& name) {
    // Organized patterns
    vector<pair<string, vector<string>>> categorized_patterns = {
        {"Mono-nucleotide Repeats (Homopolymers)", {
            "AAAA",
            "AAAAAAAA",
            "AAAAAAAAAAAA",
            "AAAAAAAAAAAAAAAAAA",
            "AAAAAAAAAAAAAAAAAAAAAAAA",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        }},
        {"Di-nucleotide Repeats (2 bp motifs)", {
            "ATAT",                // (AT)₂
            "CACACACA",            // (CA)₄
            "CACACACACACA",        // (CA)₆
            "CACACACACACACACACA",  // (CA)₉ (imperfect, odd repeat)
            "ACACACACACACACACACACACACACACACAC" // (AC)₁₆
        }},
        {"Real Data Patterns (Non-repeating/Long/Complex)", {
            "CCTG",                // 4 bp, non-repeating
            "GGAATTCC",            // 8 bp, non-repeating
            "CTCTTTGAGAAG",        // 12 bp, no clear repeat
            "TATACATACACGCACACA",  // 18 bp, no clear repeat
            "GGGAGGGCTGCTAGTCCAGGCTGT", // 24 bp, no clear repeat
            "AGGGCTACATGTGATGGATGTGGAATATAGCA" // 32 bp, non-repeating
        }},
    };

    cout << "\n=== Benchmarking on " << name << " (" << genome.size() << "bp) ===" << endl;

    for (const auto& category : categorized_patterns) {
        cout << "\n=== " << category.first << " ===" << endl;

        for (const string& pattern : category.second) {
            int pattern_len = pattern.size();

            // Print the pattern as a header
            cout << "\n" << pattern << endl;
            cout << "Algorithm               | Pattern Length | Matches | Time (ms)" << endl;
            cout << "------------------------|----------------|---------|----------" << endl;

            // Standard Boyer-Moore
            auto start = chrono::high_resolution_clock::now();
            vector<int> matches_bm = boyer_moore(genome, pattern);
            auto end = chrono::high_resolution_clock::now();
            double time_bm = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
            printf("Boyer-Moore          | %14d | %7d | %8.4f\n",
                   pattern_len, (int)matches_bm.size(), time_bm);

            // Boyer-Moore-Horspool
            start = chrono::high_resolution_clock::now();
            vector<int> matches_bmh = BMHAlgorithm(genome, pattern);
            end = chrono::high_resolution_clock::now();
            double time_bmh = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
            printf("Boyer-Moore-Horspool | %14d | %7d | %8.4f\n",
                   pattern_len, (int)matches_bmh.size(), time_bmh);

            // Turbo Boyer-Moore
            start = chrono::high_resolution_clock::now();
            vector<int> matches_turbo = TurboBM(genome, pattern);
            end = chrono::high_resolution_clock::now();
            double time_turbo = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
            printf("Turbo Boyer-Moore    | %14d | %7d | %8.4f\n",
                   pattern_len, (int)matches_turbo.size(), time_turbo);

            // Shift-Or algorithm
            start = chrono::high_resolution_clock::now();
            vector<int> matches_shiftor = ShiftOr(genome, pattern);
            end = chrono::high_resolution_clock::now();
            double time_shiftor = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
            printf("Shift-Or             | %14d | %7d | %8.4f\n",
                   pattern_len, (int)matches_shiftor.size(), time_shiftor);

            // KMP algorithm
            start = chrono::high_resolution_clock::now();
            vector<int> matches_kmp = KMP(genome, pattern);
            end = chrono::high_resolution_clock::now();
            double time_kmp = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
            printf("KMP                  | %14d | %7d | %8.4f\n",
                   pattern_len, (int)matches_kmp.size(), time_kmp);

            // Rabin-Karp algorithm
            start = chrono::high_resolution_clock::now();
            vector<int> matches_rk = RabinKarp(genome, pattern);
            end = chrono::high_resolution_clock::now();
            double time_rk = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
            printf("Rabin-Karp           | %14d | %7d | %8.4f\n",
                   pattern_len, (int)matches_rk.size(), time_rk);

            // SIMD Boyer-Moore
            start = chrono::high_resolution_clock::now();
            vector<int> matches_simdbm = SIMDBM(genome, pattern);
            end = chrono::high_resolution_clock::now();
            double time_simdbm = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
            printf("SIMD Boyer-Moore     | %14d | %7d | %8.4f\n",
                   pattern_len, (int)matches_simdbm.size(), time_simdbm);

            cout << "------------------------|----------------|---------|----------" << endl;
        }
    }
}


int main() {
    cout << "Reading DNA sequence from FASTA file..." << endl;
    string dna_sequence = read_fasta("sequence.fasta");
    if (dna_sequence.empty()) {
        cerr << "Error: Failed to read DNA sequence" << endl;
        return 1;
    }

    run_benchmark(dna_sequence, "REAL DNA SEQUENCE");
   
    return 0;
}
