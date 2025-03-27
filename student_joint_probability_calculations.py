import numpy as np

# Defining the prior probabilities
P_A_true = 0.60
P_E_true = 0.53
P_D_true = 0.57
P_I_true = 0.34
P_A_false = 0.40
P_E_false = 0.47
P_D_false = 0.43
P_I_false = 0.66

# Defining the conditional probabilities
P_M_true_E_true_I_true = 0.85
P_M_true_E_true_I_false = 0.45
P_M_false_E_true_I_true = 0.35
P_M_false_E_true_I_false = 0.75
P_M_true_E_false_I_true = 0.65
P_M_true_E_false_I_false = 0.25
P_M_false_E_false_I_true = 0.55
P_M_false_E_false_I_false = 0.75

P_S_true_E_true_D_true = 0.44
P_S_true_E_true_D_false = 0.74
P_S_false_E_true_D_true = 0.56
P_S_false_E_true_D_false = 0.26
P_S_true_E_false_D_true = 0.19
P_S_true_E_false_D_false = 0.36
P_S_false_E_false_D_true = 0.81
P_S_false_E_false_D_false = 0.64

P_F_true_M_true_S_true = 0.79
P_F_true_M_true_S_false = 0.69
P_F_false_M_true_S_true = 0.21
P_F_false_M_true_S_false = 0.31
P_F_true_M_false_S_true = 0.39
P_F_true_M_false_S_false = 0.17
P_F_false_M_false_S_true = 0.61
P_F_false_M_false_S_false = 0.83

P_G_pass_M_true_F_true_A_true = 0.92
P_G_fail_M_true_F_true_A_true = 0.08
P_G_pass_M_true_F_false_A_true = 0.58
P_G_fail_M_true_F_false_A_true = 0.42
P_G_pass_M_true_F_true_A_false = 0.46
P_G_fail_M_true_F_true_A_false = 0.54
P_G_pass_M_true_F_false_A_false = 0.38
P_G_fail_M_true_F_false_A_false = 0.62
P_G_pass_M_false_F_true_A_true = 0.46
P_G_fail_M_false_F_true_A_true = 0.54
P_G_pass_M_false_F_false_A_true = 0.16
P_G_fail_M_false_F_false_A_true = 0.84
P_G_pass_M_false_F_true_A_false = 0.28
P_G_fail_M_false_F_true_A_false = 0.72
P_G_pass_M_false_F_false_A_false = 0.05
P_G_fail_M_false_F_false_A_false = 0.95

# Initialize an empty list for joint probabilities
joint_probabilities = []

# List of all combinations for M, E, D, I
combinations = [
    ('M=True, E=True, D=True, I=True', P_M_true_E_true_I_true, P_S_true_E_true_D_true, P_F_true_M_true_S_true, P_G_pass_M_true_F_true_A_true, P_A_true, P_E_true, P_D_true, P_I_true),
    ('M=True, E=True, D=True, I=False', P_M_true_E_true_I_false, P_S_true_E_true_D_true, P_F_true_M_true_S_true, P_G_pass_M_true_F_true_A_true, P_A_true, P_E_true, P_D_true, P_I_false),
    ('M=True, E=True, D=False, I=True', P_M_true_E_true_I_true, P_S_true_E_true_D_false, P_F_true_M_true_S_true, P_G_pass_M_true_F_true_A_true, P_A_true, P_E_true, P_D_false, P_I_true),
    ('M=True, E=True, D=False, I=False', P_M_true_E_true_I_false, P_S_true_E_true_D_false, P_F_true_M_true_S_true, P_G_pass_M_true_F_true_A_true, P_A_true, P_E_true, P_D_false, P_I_false),
    ('M=True, E=False, D=True, I=True', P_M_true_E_false_I_true, P_S_true_E_false_D_true, P_F_true_M_true_S_true, P_G_pass_M_true_F_true_A_true, P_A_true, P_E_false, P_D_true, P_I_true),
    ('M=True, E=False, D=True, I=False', P_M_true_E_false_I_false, P_S_true_E_false_D_true, P_F_true_M_true_S_true, P_G_pass_M_true_F_true_A_true, P_A_true, P_E_false, P_D_true, P_I_false),
    ('M=True, E=False, D=False, I=True', P_M_true_E_false_I_true, P_S_true_E_false_D_false, P_F_true_M_true_S_true, P_G_pass_M_true_F_true_A_true, P_A_true, P_E_false, P_D_false, P_I_true),
    ('M=True, E=False, D=False, I=False', P_M_true_E_false_I_false, P_S_true_E_false_D_false, P_F_true_M_true_S_true, P_G_pass_M_true_F_true_A_true, P_A_true, P_E_false, P_D_false, P_I_false),
    ('M=False, E=True, D=True, I=True', P_M_false_E_true_I_true, P_S_true_E_true_D_true, P_F_true_M_false_S_true, P_G_pass_M_false_F_true_A_true, P_A_true, P_E_true, P_D_true, P_I_true),
    ('M=False, E=True, D=True, I=False', P_M_false_E_true_I_false, P_S_true_E_true_D_true, P_F_true_M_false_S_true, P_G_pass_M_false_F_true_A_true, P_A_true, P_E_true, P_D_true, P_I_false),
    ('M=False, E=True, D=False, I=True', P_M_false_E_true_I_true, P_S_true_E_true_D_false, P_F_true_M_false_S_true, P_G_pass_M_false_F_true_A_true, P_A_true, P_E_true, P_D_false, P_I_true),
    ('M=False, E=True, D=False, I=False', P_M_false_E_true_I_false, P_S_true_E_true_D_false, P_F_true_M_false_S_true, P_G_pass_M_false_F_true_A_true, P_A_true, P_E_true, P_D_false, P_I_false),
    ('M=False, E=False, D=True, I=True', P_M_false_E_false_I_true, P_S_true_E_false_D_true, P_F_true_M_false_S_true, P_G_pass_M_false_F_true_A_true, P_A_true, P_E_false, P_D_true, P_I_true),
    ('M=False, E=False, D=True, I=False', P_M_false_E_false_I_false, P_S_true_E_false_D_true, P_F_true_M_false_S_true, P_G_pass_M_false_F_true_A_true, P_A_true, P_E_false, P_D_true, P_I_false),
    ('M=False, E=False, D=False, I=True', P_M_false_E_false_I_true, P_S_true_E_false_D_false, P_F_true_M_false_S_true, P_G_pass_M_false_F_true_A_true, P_A_true, P_E_false, P_D_false, P_I_true),
    ('M=False, E=False, D=False, I=False', P_M_false_E_false_I_false, P_S_true_E_false_D_false, P_F_true_M_false_S_true, P_G_pass_M_false_F_true_A_true, P_A_true, P_E_false, P_D_false, P_I_false),
]

# Avoid duplicates during calculations
calculated_names = set()
counter = 1  # Initialize a counter for numbering

# Calculate and display detailed steps for joint probabilities
for combo in combinations:
    name, P_M, P_S, P_F, P_G, P_A, P_E, P_D, P_I = combo

    # Skip duplicates
    if name in calculated_names:
        continue

    # Calculate the joint probability
    joint_prob = P_G * P_F * P_S * P_M * P_A * P_E * P_D * P_I
    
    # Display each calculation step with numbering
    print(f"{counter}. Calculating for {name}:")
    print(f"  P(G={P_G}) * P(F={P_F}) * P(S={P_S}) * P(M={P_M}) * P(A={P_A}) * P(E={P_E}) * P(D={P_D}) * P(I={P_I})")
    print(f"  Joint Probability: {joint_prob}")
    
    # Append to results and mark as calculated
    joint_probabilities.append((name, joint_prob))
    calculated_names.add(name)
    counter += 1  # Increment the counter
