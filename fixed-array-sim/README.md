# Fixed-array near-field ISAC simulator

One monostatic base station serves two downlink users and illuminates one position-known target using the same signals. Two mapped point scatterers have unknown mean reflection powers. The experiment tests whether six calibration settings predict clutter under a newly designed operating beam.

## Running the MATLAB implementation

Extract the package into one folder and open that folder in MATLAB. The implementation uses base MATLAB functions and implicit expansion. The combined simulator runs on its own. Place reference_fixture.mat beside the .m file to run the numerical checks.

```matlab
test = fixed_array_isac('verify');
out = fixed_array_isac('simulate', 42, 128);
% One row per method for the first scene
disp(squeeze(out.metrics(1,:,:)))
disp(out.methods)
disp(out.columns)
```

The native simulation generates new MATLAB random samples. Identical random seeds across languages need not give identical arrays. Verification uses the same stored positions, clutter powers and independent standard complex Gaussian draws in both languages. The fixture also contains Python answers for all seven designs; the MATLAB solver recomputes them from the inputs. A discrepancy raises an error. The default simulation embeds the trained parameters.

## Physical and statistical specification

| Quantity | Starting value |
|---|---|
| Carrier wavelength and spacing | 0.01 m and half-wavelength spacing |
| Array geometry | 24 fixed Tx elements and 16 fixed Rx elements along the x axis, centered at −8 mm and +8 mm |
| Nominal positions in the array plane (m) | Target (0.035, 0.73), clutter (0.085, 0.70) and (−0.12, 0.82), users (−0.19, 0.66) and (0.20, 0.84). Each coordinate varies independently by up to 0.022 m |
| Propagation and CSI | Exact spherical distances, isotropic free-space coefficients, line-of-sight user channels known to the designer |
| Tx power and user targets | 1 W total, both user SINRs at least 2 in linear units (3.01 dB) |
| User and sensing noise powers | 6e−7 and 2e−11 in the chosen baseband convention |
| Scattering | Target power 1; clutter powers independently log-uniform from exp(−0.8) to exp(1). Powers persist from calibration through operation |
| Calibration budget | Six known Tx/Rx focus pairs, 96 independent target-free complex Gaussian snapshots per pair. Rank is checked for the two mapped powers |
| Reliability target | A simultaneous 95% confidence set over the mean clutter powers under the stated sampling model |

Positions and noise levels are illustrative simulation choices. The Tx and Rx Fraunhofer distances based on full apertures are approximately 2.65 m and 1.13 m. The model assumes narrowband propagation, stationary geometry over the design interval and negligible residual self-interference. It uses ensemble echo power as its sensing metric. No RIS, Doppler processing or detector is included.

## From observations to a beam

1. For focus pair b and scatterer j compute the complex response v_bᴴ A_j f_b. The ordinary transmit transpose in A_j = b_j a_jᵀ retains the physical propagation phase. Synthesize target-free echoes and estimate their mean energy. Normalize powers by receive noise for conditioning.
2. Fit nonnegative clutter powers by least squares. Form simultaneous gamma/chi-square intervals for the probe energies with a Bonferroni correction. Intersect the resulting linear inequalities with nonnegativity. With two unknowns the box bounds are obtained by enumerating eligible polygon vertices. An empty confidence set requests recalibration.
3. For each trial beam W, predict clutter as the sum of q_j A_j WWᴴ A_jᴴ. The conservative solver uses the componentwise upper corner of the bounding box. The receiver solves a positive-definite linear system for the whitened target response. The simulation recomputes the physical SCNR at the true powers using the same selected receive filter.
4. The initializer learns one scalar mixing weight for zero-forcing user beams and target illumination in the user-channel null space. A geometric cap preserves the user SINRs. Three unfolded gradient updates use trained positive step sizes. Each update tries up to nine successively shorter steps and accepts a normalized candidate only after checking original user SINRs and objective improvement. Rejected steps retain the prior feasible design. This first implementation uses feasible starts and step rejection in place of a general cone projection. It assumes a full-row-rank user channel and a feasible zero-forcing start. Infeasible starts raise an error; empty statistical sets are reported separately.
5. Compare true-power and plug-in runs, the conservative run with classical zero-forcing start and fixed steps, a classical geometry-aware start and fixed steps, initializer only, learned steps only, and the combination. The true-power run is a local-solver comparison. It does not certify a globally optimal SCNR. The classical geometry-aware initializer uses the analytic QoS cap and no fitted weights.

## Reproducing the Python experiment

Use Python with NumPy and SciPy. Run the following in the extracted folder.

```sh
python reference.py
python check_fixture.py
```

The reference trains the initializer by ridge regression against the best feasible mixing weight on a small grid. Layer steps are selected greedily on training scenes to maximize mean log design SCNR. Seeds 100–159 are used only for training. Seeds 1000–1127 form the independent test set. Training regenerates the MAT fixture, JSON summary and per-scene CSV. Retraining on a different scenario requires updating the MATLAB embedded coefficients and scenario configuration. The two-clutter vertex enumeration is deliberately small-scale.

## Verification and interpretation

Python checks cover four shared fixtures with a dense echo-matrix implementation, independent linear programs for the confidence box, direct user SINRs and a generalized eigenvalue reference for SCNR. The reference also checks the complex gradient with a finite difference, exact noiseless recovery, rejection of an empty calibration set and feasibility across the test split. The native MATLAB verification additionally compares the complex beams, every method's metrics and boundary cases. Python and fixture checks were executed. Native MATLAB or Octave was unavailable during preparation, so the MATLAB self-test result is pending.

The frozen Python run produced 128 feasible combined designs, no empty sets, 119 confidence sets containing the true power vector, and 127 conservative SCNR bounds below the realized true-power SCNR. These last two counts measure different events. The true-power comparison uses the analytical sensing model, not a detection experiment. The median relative Frobenius error in plug-in clutter covariance evaluated at the combined solver's operating beam was 7.11%. This check uses the simulator's true covariance; it is independent of the training scenes but is not a second physical measurement campaign.

The median true SCNR was 13.424 dB for the combined solver and 13.382 dB for the classical geometry-aware initialization and fixed updates. The median of the paired per-scene differences was only +0.0021 dB. These trials do not demonstrate a material learning advantage. No hardware latency, propagation mismatch, correlated clutter or transfer across array sizes has been validated.
