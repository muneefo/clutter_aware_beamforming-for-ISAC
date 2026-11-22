# Joint-confidence fixed-array ISAC solver

Extract all files into the same folder. In MATLAB, change into that folder:
    
    fixed_array_isac('verify_joint')
    out = fixed_array_isac('joint',42,128);
    disp(out.methods)
    median(10*log10(out.metrics(out.valid,:,2)),1)

The last line gives median true SCNR in dB for the four methods in out.methods. The verify_joint command compares four shared Python beam-design fixtures; native MATLAB execution has not been tested in the delivery environment. The separate legacy simulate and verify modes run the box formulation. Both the Python and MATLAB joint implementations specialize to the two-reflector model. Native scene generation uses each language's own random-number generator.

## Estimation and robust solver

Calibration estimates powers by nonnegative least squares. Intersect the nonnegative quadrant and the six two-sided calibration inequalities to obtain a bounded polygon. A missing polygon flags recalibration. For a fixed beam and filter, the worst clutter contribution is the maximum dot product over polygon vertices. With rank-one target response b, define R(q) as the noise-normalized receive covariance. To find the optimum filter, minimize b^H R(q)^(-1) b over the polygon. This convex function decreases with each clutter power, so a boundary minimizer exists. At a candidate q, let v=R(q)^(-1)b and let d contain the two nonnegative clutter contributions for v. First-order optimality is max_z d^T(z-q) <= 0. This certifies q is worst-case for v; v also optimizes the receive quotient at q. Together these facts certify a saddle point. Check vertices first, then solve for a zero directional derivative on an edge if needed. Normalize v to unit length. Recompute the bound explicitly over all vertices.

For transmit design, fix the desired user phases and use Dykstra's nearest-point projection onto the power ball and the user SINR cones. Each user projection modifies the beam components parallel to that user's conjugate channel. The outer loop takes three normalized gradient proposals with feasibility projection and monotonicity backtracking. It evaluates the manuscript's full joint objective. The finite-depth outer loop has no certificate of global optimality. Numerical failures raise an error. A returned beam is checked against the original constraints.

## Learning and replication

Run with Python, NumPy and SciPy. From the extracted folder:

    python check_joint.py
    python joint_solver.py

The second command refits the six coefficients and selects the three stage parameters, then writes joint_model.json, joint_fixture.mat and heldout.csv. Training seeds are 100 through 159, and held-out seeds 1000 through 1127. A bias and five features map to one scalar beam mixing fraction: log(1+qhat_1), log(1+qhat_2), the two log ZF-SINR margins and 100 times the target transmit-vector norm. The joint model uses NNLS estimates in the features. The legacy robust model used the separate box upper bounds. Candidate warm-start fractions are enumerated in reference.FRACTIONS; training labels maximize the appropriate robust SCNR over feasible candidates. Ridge fitting uses 0.001 regularization including the intercept, with no hidden layers. Three steps are selected successively from {0.015,0.045,0.12,0.32} by maximizing mean log robust SCNR, freezing previously selected stages. Dykstra iterations and line search have no trainable parameters. The provided geometry baseline uses the same beam family at fraction 0.94 with feasibility projection. Methods cross the geometric/learned starts with fixed 0.12 or trained step schedules. The learned and fixed initializers are compared directly with the same schedule in geometry_matched and combined.

Each row of heldout.csv contains one geometry/method, with SCNR and SINR in linear units. All four designs see identical observations in each geometry. Each of the 512 resulting designs was checked using an independent linear program over the original inequalities for its own beam and filter. The verification script also checks the projection against SLSQP, tests a receiver whose worst case lies between vertices, and checks a directional derivative. Timing was not benchmarked.
