# embb-urllc-spectrum slicing

Code implementing the algorithm and the benchmark of the paper "Power Minimization of Downlink Spectrum Slicing for eMBB and URLLC Users"

## Abstract
> 5G technology allows heterogeneous services to share the wireless spectrum within the same radio access network. In this context, spectrum slicing of the shared radio resources is a critical task to guarantee the performance of each service. We analyze a downlink communication serving two types of traffic: enhanced mobile broadband (eMBB) and ultra-reliable low-latency communication (URLLC). Due to the nature of low-latency traffic, the base station knows the channel state information (CSI) of the eMBB users while having statistical CSI for the URLLC users. We study the power minimization problem employing orthogonal multiple access (OMA) and non-orthogonal multiple access (NOMA) schemes. Based on this analysis, we propose a lookup table-based approach and a block coordinated descent (BCD) algorithm. We show that the BCD is optimal for the URLLC power allocation. The numerical results show that NOMA leads to lower power consumption than OMA, except when the average channel gain of the URLLC user is very high. For the latter case, the optimal approach depends on the channel condition of the eMBB user. Even when OMA attains the best performance, the gap with NOMA is negligible, showing the capability of NOMA to reduce power consumption in practically every condition. 

The paper is published in IEEE Transactions on Wireless Communications and can be found [at this link](https://ieeexplore.ieee.org/document/9831059)
An older preprint version of the work can also be found [here](https://arxiv.org/abs/2110.14544)

Also, the conference version of the paper is published in WCNC 2022 [at this link](https://ieeexplore.ieee.org/document/9831059)

The results for OMA are obtainable by running
```
oma_adaptive_preemption.py
oma_fixed_preemption.py
```
where the first reproduce the results of the journal paper while the second the conference ones.

The results for NOMA are obtainable by
```
noma_feasible_heuristic.py
noma_exhaustive_optimization
```
where the first implements the "fea" algorithm in the journal and the "heuristic" algorithm in the conference paper, while the second implements the "BCD" algorithm of the journal.   
Note that all the scripts should be run with the flag ```-r``` to save the results. 

Other flags can be used to change the standard parameters, check ```environment.py``` for the possible parameters. 

Visualization of the results can be obtained running 
```
render_results_embb.py
render_results_grid.py
reder_results_PowerOutage.py
``` 
Also in this case flag ```-r``` print the results, in both .png and .tex format, in a folder with the current date created in the home directory. 
