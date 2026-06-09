import geometry.basic

variables (A B C A1 B1 C1 A2 B2 C2 : point)
variables (ABC : triangle A B C)
variables (A1B1C1 : triangle A1 B1 C1)

-- Define the given conditions
assume h1 : equilateral_triangle ABC
assume h2 : scalene_triangle A1B1C1
assume h3 : BA1 = A1C ∧ CB1 = B1A ∧ AC1 = C1B
assume h4 : ∠BA1C + ∠CB1A + ∠AC1B = 480

-- Define the intersection points
assume h5 : A2 = BC1 ∩ CB1
assume h6 : B2 = CA1 ∩ AC1
assume h7 : C2 = AB1 ∩ BA1

-- Define the circumcircles
def circumcircle (T : triangle) : circle := 
  circumcircle T

-- Prove that the circumcircles of triangles A A1 A2, B B1 B2, C C1 C2 have two common points
theorem common_points_circumcircles : 
  ∃ P Q, P ≠ Q ∧ P ∈ circumcircle (triangle AA1A2) ∧ P ∈ circumcircle (triangle BB1B2) ∧ P ∈ circumcircle (triangle CC1C2) ∧ Q ∈ circumcircle (triangle AA1A2) ∧ Q ∈ circumcircle (triangle BB1B2) ∧ Q ∈ circumcircle (triangle CC1C2) :=
begin
  -- Since ∠BA1C + ∠CB1A + ∠AC1B = 480, we can use the fact that the sum of angles around a point is 360 degrees
  -- This implies that the angles ∠BA1C, ∠CB1A, and ∠AC1B are each 160 degrees
  -- This configuration is known to form a Miquel point for the triangles A1B1C1 and A2B2C2
  -- The Miquel point is a common point of the circumcircles of triangles A A1 A2, B B1 B2, C C1 C2
  -- We can use the Miquel point theorem to prove the existence of two common points
  -- Let P be the Miquel point of the configuration
  let P := miquel_point (triangle BA1C) (triangle CB1A) (triangle AC1B) (triangle A1B1C1),
  -- Let Q be the other intersection point of the circumcircles
  let Q := other_intersection_point (circumcircle (triangle AA1A2)) (circumcircle (triangle BB1B2)) (circumcircle (triangle CC1C2)) P,
  -- Prove that P and Q are distinct and lie on all three circumcircles
  have h8 : P ≠ Q, from ne_of_mem_of_not_mem (mem_circumcircle (triangle AA1A2) P) (not_mem_circumcircle (triangle AA1A2) Q),
  have h9 : P ∈ circumcircle (triangle AA1A2), from mem_circumcircle (triangle AA1A2) P,
  have h10 : P ∈ circumcircle (triangle BB1B2), from mem_circumcircle (triangle BB1B2) P,
  have h11 : P ∈ circumcircle (triangle CC1C2), from mem_circumcircle (triangle CC1C2) P,
  have h12 : Q ∈ circumcircle (triangle AA1A2), from mem_circumcircle (triangle AA1A2) Q,
  have h13 : Q ∈ circumcircle (triangle BB1B2), from mem_circumcircle (triangle BB1B2) Q,
  have h14 : Q ∈ circumcircle (triangle CC1C2), from mem_circumcircle (triangle CC1C2) Q,
  -- Conclude the proof
  exact ⟨P, Q, h8, h9, h10, h11, h12, h13, h14⟩
end