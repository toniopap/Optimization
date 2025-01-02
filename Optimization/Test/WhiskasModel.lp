\* Cat *\
Minimize
Total_Cost_of_Ingredients_per_can: 0.008 ing_Beef + 0.013 ing_Chicken
 + 0.001 ing_Gel + 0.01 ing_Mutton + 0.002 ing_Rice + 0.005 ing_Wheat
Subject To
FatRequirement: 0.1 ing_Beef + 0.08 ing_Chicken + 0.11 ing_Mutton
 + 0.01 ing_Rice + 0.01 ing_Wheat >= 6
FibreRequirement: 0.005 ing_Beef + 0.001 ing_Chicken + 0.003 ing_Mutton
 + 0.1 ing_Rice + 0.15 ing_Wheat <= 2
PercentagesSum: ing_Beef + ing_Chicken + ing_Gel + ing_Mutton + ing_Rice
 + ing_Wheat = 100
ProteinRequirement: 0.2 ing_Beef + 0.1 ing_Chicken + 0.15 ing_Mutton
 + 0.04 ing_Wheat >= 8
SaltRequirement: 0.005 ing_Beef + 0.002 ing_Chicken + 0.007 ing_Mutton
 + 0.002 ing_Rice + 0.008 ing_Wheat <= 0.4
End
