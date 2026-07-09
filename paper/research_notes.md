# Dataset Exploration

Dataset:
FuelCast

Vessels: 
- Poseidon (105422 samples, 65 features)
- Triton (25351 samples, 62 features)
- Ceto (43213 samples, 46 features)

Target:
- Consumer_Total_MomentaryFuel

Observations:
1. Missing values are relatively low (<3%) for most variables.
2. Ceto contains fewer features than Poseidon and Triton.
3. Weather_SwellWavePeakPeriod and Weather_WindWavePeakPeriod are completely missing in Ceto.
4. The target variable exhibits a multimodal distribution, suggesting multiple vessel operating regimes.
5. Significant propulsion-related variables are present (shaft power, shaft torque, ratation speed), which may introduce target leakage