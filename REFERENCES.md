# Part F references for author review

The numbered bibliography uses the following papers. DOI metadata were checked against Crossref or the publisher. The Liu and Adam arXiv versions were checked against their version records.

| Use in the revision | Reference | Verified identifier |
| --- | --- | --- |
| Block covariance benchmark, Fig. 12(b), and the symbol-level scope boundary | R. Liu, P. Li, M. Li, and A. L. Swindlehurst, “Clutter-Aware Integrated Sensing and Communication: Models, Methods, and Future Directions,” *Proceedings of the IEEE*, 114(1), 52–92, 2026. | [DOI 10.1109/JPROC.2026.3675476](https://doi.org/10.1109/JPROC.2026.3675476) · [arXiv:2602.10537v2](https://arxiv.org/abs/2602.10537v2) |
| Semidefinite relaxation and rank reduction in beamforming | Y. Huang and D. P. Palomar, “Rank-Constrained Separable Semidefinite Programming With Applications to Optimal Beamforming,” *IEEE Transactions on Signal Processing*, 58(2), 664–678, 2010. | [DOI 10.1109/TSP.2009.2031732](https://doi.org/10.1109/TSP.2009.2031732) |
| Quadratic-transform fractional programming | K. Shen and W. Yu, “Fractional Programming for Communication Systems—Part I: Power Control and Beamforming,” *IEEE Transactions on Signal Processing*, 66(10), 2616–2630, 2018. | [DOI 10.1109/TSP.2018.2812733](https://doi.org/10.1109/TSP.2018.2812733) |
| Low-memory, first-order semidefinite methods | A. Yurtsever, J. A. Tropp, O. Fercoq, M. Udell, and V. Cevher, “Scalable Semidefinite Programming,” *SIAM Journal on Mathematics of Data Science*, 3(1), 171–200, 2021. | [DOI 10.1137/19M1305045](https://doi.org/10.1137/19M1305045) · [arXiv:1912.02949](https://arxiv.org/abs/1912.02949) |
| Iterated subtractive transmit programs | W. Dinkelbach, “On Nonlinear Fractional Programming,” *Management Science*, 13(7), 492–498, 1967. | [DOI 10.1287/mnsc.13.7.492](https://doi.org/10.1287/mnsc.13.7.492) |
| Exact linear-fractional reformulation of the power master | A. Charnes and W. W. Cooper, “Programming with Linear Fractional Functionals,” *Naval Research Logistics Quarterly*, 9(3–4), 181–186, 1962. | [DOI 10.1002/nav.3800090303](https://doi.org/10.1002/nav.3800090303) |
| Column generation and reduced-cost pricing | G. B. Dantzig and P. Wolfe, “Decomposition Principle for Linear Programs,” *Operations Research*, 8(1), 101–111, 1960. | [DOI 10.1287/opre.8.1.101](https://doi.org/10.1287/opre.8.1.101) |
| Original policy's training method in the appendix | J. C. Spall, “Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation,” *IEEE Transactions on Automatic Control*, 37(3), 332–341, 1992. | [DOI 10.1109/9.119632](https://doi.org/10.1109/9.119632) |
| New policy optimizer in the appendix | D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” *International Conference on Learning Representations*, 2015. | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |

## Reproducibility sources

- **NR slot reference.** 3GPP TS 38.211 V18.3.0, Sections 4.2 and 4.3.2, gives 120 kHz for numerology index 3 and eight normal-CP slots in a 1 ms subframe. The duration is 125 µs by division. The standard is identified by its issuing organization's document number and version. [ETSI publication](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/18.03.00_60/ts_138211v180300p.pdf)
- **Tutorial code and matrices.** The input commit is `0f864c45b4775fbe01789b313d81f22bebf65177`. The distributed `data/covMat_BLP.mat` SHA-256 is `99744880b1a6fe5e5f8a0ef57c1738be79fb0ad28629e68db2bfe3a8df67472d`. [Pinned source](https://github.com/RangLiu0706/Clutter-Aware-ISAC-Tutorial/tree/0f864c45b4775fbe01789b313d81f22bebf65177)
