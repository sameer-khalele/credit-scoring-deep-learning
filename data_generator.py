import numpy as np
import pandas as pd

def generate_synthetic_data(samples=1000):
    """
    Generates a synthetic financial dataset for credit scoring.
    Logic: High Loan-to-Income (LTI) ratio increases the probability of default.
    """
    np.random.seed(42) # For reproducibility
    
    # Feature generation
    income = np.random.randint(20000, 150000, samples)
    loan_amount = np.random.randint(5000, 100000, samples)
    experience = np.random.randint(0, 40, samples)
    age = np.random.randint(22, 65, samples)
    
    # Engineering the LTI (Loan to Income) Ratio
    lti_ratio = loan_amount / income
    
    # Defining target (Default: 1, Healthy: 0)
    # Probability increases if LTI > 0.4 or experience is very low
    default_prob = (lti_ratio * 0.7) + (np.random.rand(samples) * 0.3)
    target = (default_prob > 0.5).astype(int)
    
    data = pd.DataFrame({
        'Income': income,
        'Loan_Amount': loan_amount,
        'LTI_Ratio': lti_ratio,
        'Experience': experience,
        'Age': age,
        'Default': target
    })
    
    return data

if __name__ == "__main__":
    df = generate_synthetic_data(1000)
    print(df.head())
    print(f"\n[INFO] Generated {len(df)} samples for training.")
