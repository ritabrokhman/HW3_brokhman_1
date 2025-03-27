# Print out the formulas used for normalization
print("1. General Normalization Formula:")
print("   P(Y | Evidence) = P(Y) / Σ P(Y)\n")
print("2. Normalizing Marginalized Probabilities:")
print("   P(E=True, D=True, I=True | Evidence) = P(E=True, D=True, I=True) / Σ P(E=True, D=True, I=True)\n")

# Function to normalize probabilities
def normalize(probabilities):
    total_sum = sum(probabilities.values())  # Compute total sum
    normalized_probabilities = {}
    calculations_log = {}  # Log the explicit calculations
    
    # Normalize each probability
    for key, value in probabilities.items():
        normalized_value = value / total_sum
        normalized_probabilities[key] = normalized_value
        calculations_log[key] = f"{value} / {total_sum} = {normalized_value}"  # Log the calculation
    
    return normalized_probabilities, total_sum, calculations_log

# Marginalized probabilities 
marginalized_probabilities_M = {
    "E=True, D=True, I=True": 0.018454673682720003,
    "E=True, D=True, I=False": 0.024298182865440006,
    "E=True, D=False, I=True": 0.023414183276880003,
    "E=True, D=False, I=False": 0.03082807730376001,
    "E=False, D=True, I=True": 0.00593008662636,
    "E=False, D=True, I=False": 0.006374586734999999,
    "E=False, D=False, I=True": 0.008476245704160001,
    "E=False, D=False, I=False": 0.00911159766,
}

marginalized_probabilities_I = {
    "M=True, E=True, D=True": 0.033967781781120004,
    "M=True, E=True, D=False": 0.04309628454048001,
    "M=True, E=False, D=True": 0.00856800651888,
    "M=True, E=False, D=False": 0.012246790481280002,
    "M=False, E=True, D=True": 0.008785074767040001,
    "M=False, E=True, D=False": 0.011145976040160002,
    "M=False, E=False, D=True": 0.0037366668424799994,
    "M=False, E=False, D=False": 0.00534105288288,
}

# Normalize the probabilities marginalized over M
print("Normalizing probabilities marginalized over M:")
normalized_M, total_sum_M, calculations_log_M = normalize(marginalized_probabilities_M)
print(f"Total sum before normalization: {total_sum_M}")
for key, value in normalized_M.items():
    print(f"{key}: {value} ({calculations_log_M[key]})")

# Normalize the probabilities marginalized over I
print("\nNormalizing probabilities marginalized over I:")
normalized_I, total_sum_I, calculations_log_I = normalize(marginalized_probabilities_I)
print(f"Total sum before normalization: {total_sum_I}")
for key, value in normalized_I.items():
    print(f"{key}: {value} ({calculations_log_I[key]})")
