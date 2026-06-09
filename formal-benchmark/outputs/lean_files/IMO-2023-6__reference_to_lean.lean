import geometry.basic

variables (A B C A1 B1 C1 A2 B2 C2 : point)
variables (ABC : triangle A B C)
variables (AA1A2 BB1B2 CC1C2 : circle)

-- Define the angles
def angle_BAC1 := angle B A C1
def angle_CBA1 := angle C B A1
def angle_ACB1 := angle A C B1

-- Given conditions
def sum_angles := angle_BAC1 + angle_CBA1 + angle_ACB1 = 480

-- Define the circumcenters
def circumcenter_A2BC := circumcenter A2 B C
def circumcenter_B2CA := circumcenter B2 C A
def circumcenter_C2AB := circumcenter C2 A B

-- Prove A1 is the circumcenter of A2BC
lemma A1_circumcenter_A2BC : circumcenter A2 B C = A1 :=
begin
  -- Using the given angle condition
  have angle_BA1C := angle_BAC1 + angle_CBA1 + angle_ACB1 = 480,
  have angle_BA2C := (180 - angle_ACB1) / 2 + (180 - angle_CBA1) / 2 + 60,
  rw angle_BA2C,
  rw angle_BA1C,
  ring,
end

-- Define the cyclic quadrilaterals
def cyclic_A2BCC1 := cyclic A2 B C C1
def cyclic_B2CAC1 := cyclic B2 C A C1
def cyclic_C2ABA1 := cyclic C2 A B A1

-- Prove B1C1B2C2 is cyclic
lemma cyclic_B1C1B2C2 : cyclic B1 C1 B2 C2 :=
begin
  -- Using the given angle condition and the fact that B1C1B2C2 is cyclic
  have angle_B1B2C := (180 - angle_ACB1) / 2 + (180 - angle_CBA1) / 2 + 60,
  have angle_B1B2C1 := (180 - angle_ACB1) / 2 + (180 - angle_CBA1) / 2 + 60,
  rw angle_B1B2C,
  rw angle_B1B2C1,
  ring,
end

-- Define the radical axis theorem
def radical_axis_X := radical_axis A1 A2 B1 B2 C1 C2

-- Prove the radical axis theorem
lemma radical_axis_X_concurrence : radical_axis A1 A2 B1 B2 C1 C2 = point X :=
begin
  -- Using the fact that A1, B1, C1 are circumcenters and the given angle condition
  have angle_B1B2C := (180 - angle_ACB1) / 2 + (180 - angle_CBA1) / 2 + 60,
  have angle_B1B2C1 := (180 - angle_ACB1) / 2 + (180 - angle_CBA1) / 2 + 60,
  rw angle_B1B2C,
  rw angle_B1B2C1,
  ring,
end

-- Define the point Y
def point_Y := isogonal_conjugate (intersection (line A A2) (line B B2)) (intersection (line B B2) (line C C2))

-- Prove the point Y has equal power with respect to the circles
lemma equal_power_Y : power A AA1A2 = power B BB1B2 = power C CC1C2 :=
begin
  -- Using the fact that A1, B1, C1 are circumcenters and the given angle condition
  have angle_B1B2C := (180 - angle_ACB1) / 2 + (180 - angle_CBA1) / 2 + 60,
  have angle_B1B2C1 := (180 - angle_ACB1) / 2 + (180 - angle_CBA1) / 2 + 60,
  rw angle_B1B2C,
  rw angle_B1B2C1,
  ring,
end

-- Conclusion
lemma common_points_circles : AA1A2 ∩ BB1B2 ∩ CC1C2 = {X, Y} :=
begin
  -- Using the fact that X and Y have equal power with respect to the circles
  have power_X := power A AA1A2 = power B BB1B2 = power C CC1C2,
  have power_Y := power A AA1A2 = power B BB1B2 = power C CC1C2,
  rw power_X,
  rw power_Y,
  exact radical_axis_X_concurrence,
end