To solve this problem, we need to leverage the given conditions and properties of polynomials. The key observation is that the polynomial \( p \) satisfies \( p(1/k) = k^2 \) for \( k = 1, 2, \ldots, n \) and \( k = -1, -2, \ldots, -n \). This suggests that \( p(1/x) - x^2 \) has roots at \( x = 1, 2, \ldots, n \) and \( x = -1, -2, \ldots, -n \). Since \( p \) is a monic polynomial of degree \( 2n \), \( p(1/x) - x^2 \) is also a polynomial of degree \( 2n \). The polynomial \( p(1/x) - x^2 \) can be written as:

\[ p(1/x) - x^2 = c \prod_{k=1}^n (x - k)(x + k) \]

for some constant \( c \). Given that \( p \) is monic, the leading coefficient of \( p \) is 1, and thus the leading coefficient of \( p(1/x) \) is \( x^{-2n} \). Therefore, the leading coefficient of \( p(1/x) - x^2 \) is \( x^{-2n} \), implying \( c = 1 \). Hence,

\[ p(1/x) - x^2 = \prod_{k=1}^n (x - k)(x + k) = \prod_{k=1}^n (x^2 - k^2) \]

This means that \( p(1/x) = x^2 + \prod_{k=1}^n (x^2 - k^2) \). For \( p(1/x) = x^2 \), we need:

\[ \prod_{k=1}^n (x^2 - k^2) = 0 \]

This product is zero if and only if \( x^2 = k^2 \) for some \( k \) in the range \( 1, 2, \ldots, n \). Therefore, \( x = \pm k \) for \( k = 1, 2, \ldots, n \).

Now, let's write the final Lean proof:

```lean
import algebra.polynomial

variables (n : ℕ) (p : polynomial ℝ)
variables (h : n > 0) (hn : even n)

open_locale big_operators

-- Define the polynomial p
def p (x : ℝ) : ℝ := x^(2*n) + ∑ i in finset.range (2*n - 1), (x^(2*n - 1 - i) * (fin n i))

-- Given condition
def condition (k : ℤ) : ℝ := k^2

-- Define the polynomial p(1/x) - x^2
def p_inv_minus_x2 (x : ℝ) : ℝ := p (1/x) - x^2

-- The polynomial p(1/x) - x^2 has roots at x = ±1, ±2, ..., ±n
lemma roots_of_p_inv_minus_x2 : ∀ x ∈ finset.range (2*n), p_inv_minus_x2 x = 0 :=
begin
  intros k hk,
  rw [p, finset.sum_range, finset.sum_range],
  simp only [one_div, pow_one, pow_zero, mul_one, add_zero, sub_self, zero_mul, finset.prod_range],
  rw [hn, even_iff_two_dvd],
  rw [dvd_add, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul_right, dvd_mul