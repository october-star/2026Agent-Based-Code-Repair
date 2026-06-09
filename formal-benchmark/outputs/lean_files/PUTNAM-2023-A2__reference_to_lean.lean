```lean
import algebra.polynomial

variables (n : ℕ) [even n] [fact (n > 0)]
variables (p : polynomial ℝ) [monic p] [degree p (2 * n)]

def q (x : ℝ) : ℝ := x^(2 * n + 2) - p (1 / x) * x^(2 * n)

lemma root_q (k : ℕ) (hk : 1 ≤ k ∧ k ≤ n) : q (real.of_int k) = 0 :=
begin
  rw q,
  simp only [p.pow, p.coeff_zero, p.coeff_one, p.coeff_nat_degree, one_smul, int.coe_of_int_pow],
  ring,
end

lemma q_factorization : ∃ a b : ℝ, ∃ c : polynomial ℝ, 
  c = (x^2 + a * x + b) * (x^2 - 1) * (x^2 - 4) * ... * (x^2 - n^2) ∧
  (x^2 + a * x + b).monic ∧
  (x^2 + a * x + b).degree 2 ∧
  (x^2 + a * x + b).coeff 0 = -(n!)^2 ∧
  (x^2 + a * x + b).coeff 1 = 0 :=
begin
  use [0, -(n!)^(-2)],
  use (x^2 - (n!)^(-2)),
  simp only [monic_def, degree_monomial, degree_pow, degree_mul, degree_one, degree_zero, degree_add],
  simp only [coeff_mul, coeff_one, coeff_zero, coeff_pow],
  ring,
end

lemma final_roots : ∃ x : ℝ, p (1 / x) = x^2 ∧ x ≠ ±1 ∧ x ≠ ±2 ∧ ... ∧ x ≠ ±n ∧ x = ±(1 / (n : ℝ)) :=
begin
  have hq : ∀ k : ℕ, 1 ≤ k → k ≤ n → q (real.of_int k) = 0,
  { intros k hk1 hk2, exact root_q k ⟨hk1, hk2⟩ },
  have hq_factor : ∃ a b : ℝ, ∃ c : polynomial ℝ, 
    c = (x^2 + a * x + b) * (x^2 - 1) * (x^2 - 4) * ... * (x^2 - n^2) ∧
    (x^2 + a * x + b).monic ∧
    (x^2 + a * x + b).degree 2 ∧
    (x^2 + a * x + b).coeff 0 = -(n!)^2 ∧
    (x^2 + a * x + b).coeff 1 = 0,
  { exact q_factorization },
  obtain ⟨a, b, c, hc⟩ := hq_factor,
  have hcoeff0 : c.coeff 0 = -(n!)^2,
  { rw hc.1, simp only [coeff_mul, coeff_one, coeff_zero, coeff_pow], ring },
  have hcoeff1 : c.coeff 1 = 0,
  { rw hc.1, simp only [coeff_mul, coeff_one, coeff_zero, coeff_pow], ring },
  have hmonic : c.monic,
  { rw hc.2, simp only [monic_def, degree_monomial, degree_pow, degree_mul, degree_one, degree_zero, degree_add], },
  have hdegree2 : c.degree 2,
  { rw hc.3, simp only [degree_monomial, degree_pow, degree_mul, degree_one, degree_zero, degree_add], },
  have hcoeff0_eq : c.coeff 0 = -(n!)^2,
  { rw hcoeff0, },
  have hcoeff1_eq : c.coeff 1 = 0,
  { rw hcoeff1, },
  have hmonic_eq : c.monic,
  { rw hmonic, },
  have hdegree2_eq : c.degree 2,
  { rw hdegree2, },
  have hq_eq : q = c,
  { rw [q, c], simp only [coeff_mul, coeff_one, coeff_zero, coeff_pow], ring },
  have hq_eq0 : ∀ x : ℝ, x ≠ 0 → q x = 0 → p (1 / x) = x^2,
  { intros x hx qx,
    rw q at qx,
    simp only [qx, p.pow, p.coeff_zero, p.coeff_one, p.coeff_nat_degree, one_smul, int.coe_of_int_pow, x_pow],
    ring, },
  have hroots : ∃ x : ℝ, x ≠ ±1 ∧ x ≠ ±2 ∧ ... ∧ x ≠ ±n ∧ p (1 / x) = x^2,
  { use [1 / (n : ℝ)],
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, hq_eq0, one_div, int.coe_of_int_pow, x_pow],
    ring,
    simp only [hcoeff0_eq, hcoeff1_eq, hmonic_eq, hdegree2_eq, hq_eq, h