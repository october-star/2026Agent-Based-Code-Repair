import data.nat.basic

variables (n : ℕ) (h : n ≥ 3)

def max_interesting_pairs (n : ℕ) : ℕ := 
  if n = 3 then 2 else (n - 2) * (n - 1) / 2

theorem max_interesting_pairs_theorem : 
  ∀ n : ℕ, n ≥ 3 → max_interesting_pairs n = (n - 2) * (n - 1) / 2 :=
begin
  intros n h,
  cases n with n,
  { simp [max_interesting_pairs] },
  { 
    have h' : n - 2 ≥ 1, from le_trans (sub_le_sub_right h) (by linarith),
    have h'' : (n - 2) * (n - 1) / 2 = (n - 2) * (n - 1) / 2, from congr_arg (λ x, x / 2) (by linarith),
    rw [max_interesting_pairs, if_pos h'],
    exact h''
  }
end

end