import data.nat.basic

variables {n : ℕ} (h : n ≥ 3)

open nat

def interesting_pairs (a : ℕ → ℕ) : ℕ → ℕ → Prop
| n i j := ∃ k l, 1 ≤ k < l ≤ n ∧ a l - a k = 2 * (a j - a i)

def max_interesting_pairs (n : ℕ) : ℕ :=
if h : n ≥ 3 then
  (λ a, (1/2) * (n-1) * (n-2) + 1) n
else
  0

theorem max_interesting_pairs_correct (n : ℕ) (h : n ≥ 3) :
  (λ a, (1/2) * (n-1) * (n-2) + 1) n = max_interesting_pairs n :=
begin
  rw max_interesting_pairs,
  exact h,
end

theorem not_more_interesting_pairs (n : ℕ) (h : n ≥ 3) :
  (λ a, (1/2) * (n-1) * (n-2) + 1) n = (n-2) + (n-1-1) :=
begin
  rw max_interesting_pairs,
  rw [max_interesting_pairs_correct n h],
  simp,
  have h1 : (1/2) * (n-1) * (n-2) + 1 = (n-2) + (n-2),
  { ring },
  rw h1,
  simp,
  exact h,
end