Three plots from the completed MATLAB Monte Carlo experiment

Open isac_monte_carlo_figures.pdf to view all three figures. Individual
PDFs are vector graphics for the manuscript and PNGs support quick viewing.

1. runtime shows median online solver time on a logarithmic axis.
2. sensing_quality shows median true-SCNR gap (AO minus method, dB) and
   counts meeting the one-sided 0.1 dB requirement over feasible AO references.
3. learning_effect shows median paired learned-minus-fixed true SCNR in
   thousandths of a dB, with 95% within-setting percentile bootstrap intervals.

The summary.csv and paired_intervals.csv tables were transcribed from the
printed full-run results, with precision as printed. They contain all 30
method/budget summary rows and 9 paired ablation rows. Timing repeats are
not independent channel samples. The same 24 seeds are reused across loads.

To display all three figures in MATLAB, extract this ZIP, open its folder,
then run the following:
    figs = plot_full_isac_results();

To use the original full-precision CSVs on your PC, run:
    figs = plot_full_isac_results(R.folder);

The script opens three visible MATLAB figures and writes a PDF and PNG for
each to a publication_figures subfolder of the chosen data directory.
It only reads aggregate CSVs. No optimization is performed.
