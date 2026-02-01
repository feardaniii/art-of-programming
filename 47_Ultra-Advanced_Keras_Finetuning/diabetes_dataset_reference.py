# ============================================================================
# DIABETES DATASET FEATURE REFERENCE
# ============================================================================
# Save this file - it's your forever reference for sklearn's diabetes dataset
# ============================================================================

DIABETES_FEATURES = {
    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE    FULL NAME                    UNIT        GOOD DIRECTION
    # ─────────────────────────────────────────────────────────────────────────

    'age': {
        'name': 'Age',
        'unit': 'years',
        'avg_real': 48,
        'direction': 'neutral',  # Just a demographic
        'decode': lambda x: f"{48 + (x * 200):.0f} years"  # Approximate
    },

    'sex': {
        'name': 'Sex',
        'unit': 'binary',
        'values': {0.05: 'Male', -0.04: 'Female'},
        'direction': 'neutral',
        'decode': lambda x: 'Male' if x > 0 else 'Female'
    },

    'bmi': {
        'name': 'Body Mass Index',
        'unit': 'kg/m²',
        'avg_real': 26,
        'direction': 'lower_better',  # ⬇️ = healthier
        'ranges': {
            'underweight': '<18.5',
            'normal': '18.5-25',
            'overweight': '25-30',
            'obese': '>30'
        },
        'decode': lambda x: f"{26 + (x * 80):.1f} kg/m²"
    },

    'bp': {
        'name': 'Blood Pressure (MAP)',
        'unit': 'mmHg',
        'avg_real': 94,
        'direction': 'lower_better',  # ⬇️ = healthier
        'decode': lambda x: f"{94 + (x * 200):.0f} mmHg"
    },

    's1': {
        'name': 'Total Cholesterol',
        'unit': 'mg/dL',
        'avg_real': 200,
        'direction': 'lower_better',  # ⬇️ = healthier
        'ranges': {
            'desirable': '<200',
            'borderline': '200-239',
            'high': '>240'
        },
        'decode': lambda x: f"{200 + (x * 800):.0f} mg/dL"
    },

    's2': {
        'name': 'LDL (Bad Cholesterol)',
        'unit': 'mg/dL',
        'avg_real': 115,
        'direction': 'lower_better',  # ⬇️ = healthier (clogs arteries)
        'decode': lambda x: f"{115 + (x * 700):.0f} mg/dL"
    },

    's3': {
        'name': 'HDL (Good Cholesterol)',
        'unit': 'mg/dL',
        'avg_real': 50,
        'direction': 'higher_better',  # ⬆️ = healthier (OPPOSITE!)
        'note': '⚠️ ONLY FEATURE WHERE HIGHER IS BETTER',
        'decode': lambda x: f"{50 + (x * 250):.0f} mg/dL"
    },

    's4': {
        'name': 'TC/HDL Ratio',
        'unit': 'ratio',
        'avg_real': 4.0,
        'direction': 'lower_better',  # ⬇️ = healthier
        'decode': lambda x: f"{4.0 + (x * 20):.1f}"
    },

    's5': {
        'name': 'Triglycerides (log)',
        'unit': 'mg/dL',
        'avg_real': 150,
        'direction': 'lower_better',  # ⬇️ = healthier
        'decode': lambda x: f"{150 + (x * 1000):.0f} mg/dL"
    },

    's6': {
        'name': 'Blood Sugar (Glucose)',
        'unit': 'mg/dL',
        'avg_real': 91,
        'direction': 'lower_better',  # ⬇️ = healthier
        'ranges': {
            'normal': '<100',
            'prediabetic': '100-125',
            'diabetic': '>126'
        },
        'decode': lambda x: f"{91 + (x * 280):.0f} mg/dL"
    }
}

# ============================================================================
# QUICK REFERENCE TABLE (copy-paste friendly)
# ============================================================================
"""
┌─────────┬──────────────────────────┬─────────┬────────────────────────────┐
│ Feature │ What It Is               │ Unit    │ Risk Direction             │
├─────────┼──────────────────────────┼─────────┼────────────────────────────┤
│ age     │ Age                      │ years   │ --                         │
│ sex     │ Sex                      │ binary  │ --                         │
│ bmi     │ Body Mass Index          │ kg/m²   │ ⬆️ Higher = WORSE (obese)  │
│ bp      │ Blood Pressure           │ mmHg    │ ⬆️ Higher = WORSE          │
│ s1      │ Total Cholesterol        │ mg/dL   │ ⬆️ Higher = WORSE          │
│ s2      │ LDL (Bad Cholesterol)    │ mg/dL   │ ⬆️ Higher = WORSE          │
│ s3      │ HDL (Good Cholesterol)   │ mg/dL   │ ⬇️ LOWER = WORSE ⚠️        │
│ s4      │ Cholesterol/HDL Ratio    │ ratio   │ ⬆️ Higher = WORSE          │
│ s5      │ Triglycerides            │ mg/dL   │ ⬆️ Higher = WORSE          │
│ s6      │ Blood Sugar (Glucose)    │ mg/dL   │ ⬆️ Higher = WORSE          │
└─────────┴──────────────────────────┴─────────┴────────────────────────────┘

⚠️  S3 (HDL) IS THE EXCEPTION: It's "good" cholesterol, so LOW = BAD
    When you see s3: -0.03, that means LOW HDL = cardiovascular risk!
"""

# ============================================================================
# NORMALIZED VALUE DECODER
# ============================================================================
"""
The dataset is pre-normalized. Here's how to read values:

    Value    │  Meaning
   ──────────┼─────────────────────────
    0.00     │  Average (mean)
   +0.05     │  ~1 std dev ABOVE average
   -0.05     │  ~1 std dev BELOW average
   +0.10     │  ~2 std dev above (high)
   -0.10     │  ~2 std dev below (low)
"""


# ============================================================================
# HELPER FUNCTION: Decode a patient's values
# ============================================================================
def decode_patient(values, feature_names=None):
    """
    Convert normalized values back to human-readable format.

    Usage:
        patient = [0.05, 0.05, 0.06, 0.02, 0.03, 0.02, -0.03, 0.05, 0.04, 0.05]
        decode_patient(patient)
    """
    if feature_names is None:
        feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

    print("Patient Profile:")
    print("-" * 60)

    for name, val in zip(feature_names, values):
        feat = DIABETES_FEATURES[name]
        decoded = feat['decode'](val)
        direction = feat.get('direction', 'neutral')

        # Flag concerning values
        flag = ""
        if direction == 'lower_better' and val > 0.03:
            flag = "⚠️ HIGH"
        elif direction == 'higher_better' and val < -0.02:
            flag = "⚠️ LOW"

        print(f"  {name:5} │ {val:+.3f} │ {decoded:20} │ {flag}")

    print("-" * 60)


# ============================================================================
# DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DIABETES DATASET FEATURE REFERENCE")
    print("=" * 70)
    print(__doc__ if __doc__ else "")

    # Print the quick reference
    print("""
┌─────────┬──────────────────────────┬─────────┬────────────────────────────┐
│ Feature │ What It Is               │ Unit    │ Risk Direction             │
├─────────┼──────────────────────────┼─────────┼────────────────────────────┤
│ age     │ Age                      │ years   │ --                         │
│ sex     │ Sex                      │ binary  │ --                         │
│ bmi     │ Body Mass Index          │ kg/m²   │ ⬆️ Higher = WORSE (obese)  │
│ bp      │ Blood Pressure           │ mmHg    │ ⬆️ Higher = WORSE          │
│ s1      │ Total Cholesterol        │ mg/dL   │ ⬆️ Higher = WORSE          │
│ s2      │ LDL (Bad Cholesterol)    │ mg/dL   │ ⬆️ Higher = WORSE          │
│ s3      │ HDL (Good Cholesterol)   │ mg/dL   │ ⬇️ LOWER = WORSE ⚠️        │
│ s4      │ Cholesterol/HDL Ratio    │ ratio   │ ⬆️ Higher = WORSE          │
│ s5      │ Triglycerides            │ mg/dL   │ ⬆️ Higher = WORSE          │
│ s6      │ Blood Sugar (Glucose)    │ mg/dL   │ ⬆️ Higher = WORSE          │
└─────────┴──────────────────────────┴─────────┴────────────────────────────┘

⚠️  S3 (HDL) IS THE EXCEPTION: Low = Bad (it's the "good" cholesterol)
""")

    print("\n" + "=" * 70)
    print("EXAMPLE: Decoding a patient")
    print("=" * 70 + "\n")

    example_patient = [0.05, 0.05, 0.06, 0.02, 0.03, 0.02, -0.03, 0.05, 0.04, 0.05]
    decode_patient(example_patient)

    print("""
CLINICAL TRANSLATION:
  "58-year-old male with obesity, low HDL, and pre-diabetic glucose"
  → HIGH RISK for diabetes progression
""")