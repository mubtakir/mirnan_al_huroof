# Quantum Wave Simulation and 7-Oscillator SPICE Netlist for XOR

This plan details the design and implementation of:
1. The new Julia module [prnn_quantum_wave.jl](file:///C:/Users/allmy/Desktop/aaa/mirnan_julia/prnn_toy/prnn_quantum_wave.jl) demonstrating three wave-based quantum computing metaphors: Double-Slit (Phase Bifurcation), Wave Annihilation (NOT/Antipode), and Entanglement/Collapse.
2. The 7-oscillator SPICE Netlist model mapping the XOR resolution using positive/negative coupling parameters.

## Proposed Changes

### [PRNN Toy Suite]

#### [NEW] [prnn_quantum_wave.jl](file:///C:/Users/allmy/Desktop/aaa/mirnan_julia/prnn_toy/prnn_quantum_wave.jl)
Implement the three proposed physical experiments in Julia:
- **Double-Slit (Phase Bifurcation)**: Create a wave $z_0$ and split it into $z_1 + z_2$ (phase-shifted by $\pi/2$). Allow them to propagate in a grid or coupled system to display spatial amplitude interference fringes.
- **Wave Annihilation (NOT/Antipode Gate)**: Initialize two phase-locked oscillators, then invert one's phase (multiply by $-1$ or shift by $\pi$) and simulate how they decay to zero amplitude.
- **Entanglement & Collapse**: Couple two Stuart-Landau oscillators $A$ and $B$ strongly ($K_{AB}$). Let them evolve into a phase-entangled state. Apply a clamp (measurement) on $A$ and show that $B$ collapses instantly to the corresponding phase.

#### [NEW] [prnn_xor_7osc.cir](file:///C:/Users/allmy/Desktop/aaa/mirnan_julia/prnn_toy/prnn_xor_7osc.cir)
Write the complete 7-oscillator SPICE netlist representing the XOR gate:
- Inputs: `NODE1`, `NODE2`
- Bias reference: `NODE3` (clamped to 0)
- Hidden layer: `NODE4`, `NODE5`, `NODE6`
- Output: `NODE7` (free)
- Coupling weights: Programmed using VCCS (`G` elements) representing the trained coupling matrix $K_{ij}$ from `prnn_toy.jl`.

## Verification Plan

### Automated Tests
- Run the new Julia file using:
  ```powershell
  julia --project=C:/Users/allmy/Desktop/aaa/mirnan_julia C:/Users/allmy/Desktop/aaa/mirnan_julia/prnn_toy/prnn_quantum_wave.jl
  ```
- Verify that all three experiments exit with success and output clear diagnostic tables or ASCII plots of the states.
