import numpy as np

# Print out the formulas used for the calculations
print("1. General Marginalization Formula:")
print("   P(Y) = ΣX P(X, Y)")
print("2. Marginalizing Over M (Midterm Score):")
print("   P(E=True, D=True, I=True) = P(M=True, E=True, D=True, I=True) + P(M=False, E=True, D=True, I=True)\n")
print("3. Marginalizing Over I (Intelligence):")
print("   P(E=True, D=True) = ΣI [P(I=True, E=True, D=True) + P(I=False, E=True, D=True)]\n")

# Joint probabilities
joint_probabilities = [
    ("M=True, E=True, D=True, I=True", 0.016752028898880004),
    ("M=True, E=True, D=True, I=False", 0.017215752882240004),
    ("M=True, E=True, D=False, I=True", 0.021253969679520004),
    ("M=True, E=True, D=False, I=False", 0.021842314860960006),
    ("M=True, E=False, D=True, I=True", 0.00490551668568),
    ("M=True, E=False, D=True, I=False", 0.0036624898331999996),
    ("M=True, E=False, D=False, I=True", 0.007011763462080002),
    ("M=True, E=False, D=False, I=False", 0.005235027019200001),
    ("M=False, E=True, D=True, I=True", 0.0017026447838399998),
    ("M=False, E=True, D=True, I=False", 0.007082429983200001),
    ("M=False, E=True, D=False, I=True", 0.0021602135973600002),
    ("M=False, E=True, D=False, I=False", 0.008985762442800003),
    ("M=False, E=False, D=True, I=True", 0.00102456994068),
    ("M=False, E=False, D=True, I=False", 0.0027120969017999995),
    ("M=False, E=False, D=False, I=True", 0.00146448224208),
    ("M=False, E=False, D=False, I=False", 0.0038765706407999996),
]

# Function to marginalize over a variable
def marginalize(variable, joint_probabilities):
    marginal_probabilities = {}
    additions_log = {}  # Log additions for explanation
    
    # Loop through all joint probabilities
    for joint_entry in joint_probabilities:
        description, probability = joint_entry
        
        # Parse the variables in the description
        variables = {var.split("=")[0]: var.split("=")[1] for var in description.split(", ")}
        
        # Remove the marginalized variable and create a new key
        reduced_key = ", ".join(f"{key}={value}" for key, value in variables.items() if key != variable)
        
        # Add the probability to the reduced key
        if reduced_key in marginal_probabilities:
            marginal_probabilities[reduced_key] += probability
            additions_log[reduced_key].append(probability)
        else:
            marginal_probabilities[reduced_key] = probability
            additions_log[reduced_key] = [probability]
    
    return marginal_probabilities, additions_log

# Marginalize over "M" (Midterm Score)
marginalized_over_M, additions_log_M = marginalize("M", joint_probabilities)
print("Marginalized over M:")
for key, value in marginalized_over_M.items():
    print(f"{key}: {value}")
    print(f"  Numbers added: {additions_log_M[key]}")

# Marginalize over "I" (Intelligence)
marginalized_over_I, additions_log_I = marginalize("I", joint_probabilities)
print("\nMarginalized over I:")
for key, value in marginalized_over_I.items():
    print(f"{key}: {value}")
    print(f"  Numbers added: {additions_log_I[key]}")
