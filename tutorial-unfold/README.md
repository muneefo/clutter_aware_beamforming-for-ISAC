# MATLAB comparison with Liu et al., Section VI-A / Fig. 12(b)

This package calls the authors' own alternating-optimization implementation as the reference and reuses their kernel, MVDR, clutter-adjoint, and SCNR functions. The new solver unrolls beam-space updates derived from Eq. (124). It is a finite-depth inexact approach. No runtime or SCNR improvement on the tutorial's dataset has been verified before packaging.

## Run

Get https://github.com/RangLiu0706/Clutter-Aware-ISAC-Tutorial and configure MATLAB R2022a or later, CVX, and a supported SDP solver. The upstream solve_TX_Dinkelbach.m selects MOSEK internally. The authors' README explains the SDPT3/SeDuMi alternatives. The package also requires the upstream function directory and data/covMat_BLP.mat. If that file is missing, generate the parameters and the kernel file using the authors' param_basic.m and Vcc_Reta_esti.m. The upstream estimation script currently uses a Windows-style relative filename in its final save call. On Linux, use fullfile(root,'data','covMat_BLP.mat') as the first argument of that save call to write in the intended directory. This offline calibration can be expensive and uses the toolboxes required by the authors' scripts.

Extract this package. From the extracted folder:
    
    repo = '/absolute/path/to/Clutter-Aware-ISAC-Tutorial';
    compare_tutorial_unfolding(repo,struct('selfTest',true));
    out = compare_tutorial_unfolding(repo);
    disp(out.trials)

The self-test uses a small synthetic matrix fixture and the authors' helper functions; it requires neither CVX nor the large covariance file. It checks direct physical and kernel formulas, received SCNR, beam gradient, SINR, feasibility and a directional derivative through the policy. The Python fixture generator has been executed. Native MATLAB and CVX execution require the user's installation.

Default Fig. 12(b) operating point: K=4, gamma=8 dB, 100 W, 32 subcarriers, 16 TX and 16 RX antennas, the figure script's target gain, and the same R_eta in design and evaluation. Like that script this uses Vcc_hat for design and Vcc for evaluating true SCNR. Only communication channels and initialization vary between trials. This evaluates channel generalization within one saved clutter environment. Defaults of 24 training channels, three passes, three unfolded layers, and four held-out channels are a pilot. To run a paired 100-scene study at this operating point:

    opts = struct('nTest',100,'folder','unfold_100_results');
    out = compare_tutorial_unfolding(repo,opts);

Change K and gammaDB to run another operating point. Set folder to retain each run separately. Results will also depend on machine, CVX backend, and MATLAB/BLAS configuration.

## Algorithms and matching

- Original AO uses solve_BLP_AO, including Dinkelbach-SDP, the repository's tolerances, and its own beam recovery.
- Fixed control and learned method use identical initialization repair, MVDR, gradient, feasibility correction, accept/reject rule and depth. Both replay the original random unstructured initial beam draw using the same RNG seed. Only the beam-space methods repair this draw against the user constraints before layer one. The repair is timed. A failed zero-forcing anchor is logged as an algorithm failure. It does not certify infeasibility of the design problem.
- For W, form Q=WW^H and compute the same receive beam as the original solver at Q. Use the estimated-kernel SCNR as eta. The direction for carrier n is (A_target,n - eta A_clutter,n)W_n. Normalize the collection of directions by its Frobenius norm and scale by sqrt(P). Clip candidate power, rotate each user beam to positive desired phase, then, when necessary, find the largest feasible fraction along a line from a feasible zero-forcing anchor by bisection. The fixed-phase SINR constraints are second-order cones. This is a feasible segment correction, not the closest-point projection or an exact Dinkelbach solve. Reject a step if estimated-kernel SCNR falls. Learning never uses the true evaluation kernel.
- The fixed step maximizes mean final log design SCNR on the training split over [0.03, 0.1, 0.3, 0.6, 1]. The neural policy uses a shared 3-to-8-to-1 tanh MLP with a bounded sigmoid output in [0.01,1]. Inputs are current post-filter cold-clutter fraction, tanh(log worst-user SINR margin ratio), and relative stage number. It has 41 trainable parameters. Adam maximizes final log design SCNR. Online inference needs only the forward MLP. Offline training uses central differences in each stage logit through the remaining solver trajectory and the analytic network Jacobian. Training is approximate around branch changes. There is no dependency on Deep Learning Toolbox.
- Each method receives identical channel draws and clutter estimates on a test scene. Seeds for calibration/training the policy, tuning the step, warm-up/profiling, and held-out evaluation are handled by the driver; the saved clutter kernels themselves are supplied by the upstream project. The fixed step and policy use the same training split. Training time is measured separately.

## Interpret results

- trials.csv reports online seconds, minimum SINR margin, total power, feasibility, true and design SCNR. AO also reports paperCovTrueSCNRdB using its returned covariances as in Fig. 12(b). The three-way comparison uses physically recovered beam matrices and the as-returned combiners; with rank truncation, covariance and recovered-beam scores can differ.
- gapToAOdB = feasible AO recovered-beam SCNR minus the method's final feasible SCNR, in dB, on the same scene. Report SCNR *and* feasibility across all attempts. A negative value is possible for nonconvex methods. Empty values indicate an unavailable comparison; inspect the message field.
- layers.csv gives prefix timing and true SCNR for every unfolded layer. firstHitSeconds is an **offline diagnostic**, the first feasible layer reaching the final feasible AO SCNR minus qualityGapDB (0.1 dB by default). speedAtQuality divides the full reference solve time by that prefix time. This uses ground truth for retrospective evaluation and supplies no deployable early-stopping rule. It does not measure when AO first crosses a separate common target.
- Online time includes solver initialization, correction, receiver updates, network inference where used, and beam recovery for AO. It excludes training, channel/measurement generation, reporting, and independent post-solve scoring. Methods run in rotating order after warm-up. One additional AO run is profiled separately; baseline_profile.csv contains inclusive times and call counts. Child times overlap parent times. Trained weights, configuration, training seeds, and offline training time are in trained_model.mat.
