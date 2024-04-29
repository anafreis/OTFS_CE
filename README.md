# LSTM-Based Channel Estimation For OTFS Modulation

This project presents codes for the LS-LSTM-NN channel estimation affected by HPA impairments proposed for OTFS channels in "LSTM-Based Time-Frequency Domain Channel Estimation for OTFS Modulation".

This project presents:

1) The SFFT-based OTFS Tx-Rx implementation with the TCE [1] channel estimation: Communication scenario provided in Matlab
2) The full training and testing phases LS-LSTM-NN channel estimators proposed: LSTM network implementation provided in Python
----------------------------------------------------------------------------------------------------------------------
The following instructions will guide the execution:
1) Matlab_Codes/main_OTFS_CE_NLD.m: Present the main simulation file. The user needs to define the simulation parameters (Speed, channel model, modulation order, HPA IBO, [...]). This file will obtain the results for the benchmark TCE estimation [1] and save the datasets for the initial LS estimation used in the proposed method.
2) Matlab_Codes/Datasets_Generation_NLD: The testing and training datasets are generated in the path Python_Codes - NLD
3) Python_Codes/LSTM_Training.py:  The LSTM-NN training is performed by employing the training dataset. 500 models are saved and considered as average (in order to obtain reproducible results) in the next step.
4) Python_codes/LSTM_Avg_Model: The latest 50 trained models are averaged.
5) Python_Codes/LSTM_Testing.py: The LSTM-NN model is tested by considering the testing datasets, and the results are saved in .mat files.
6) Matlab_Codes/Results_Processing_NLD: Results from Python are processed and saved
7) Matlab_Codes/Plotting_[...]: Results are plotted in the different metrics

Additional files:
1) Matlab_Codes/OTFS_functions.m: Includes functions for SFFT-based OTFS transmission.
2) Matlab_Codes/Channel_functions.m: Includes the pre-defined vehicular channel models [2] for different mobility conditions.
3) Matlab_codes/[NLD;hpa_saleh;function_HPA_WONG5_Poly;charac_hpa;]: Files related to the Memoryless Polynomial HPA described in [3].
4) Matlab_codes/circulant: Used to obtain the circulant matrix in the TCE method.
----------------------------------------------------------------------------------------------------------------------
[1] P. Raviteja, K. T. Phan, and Y. Hong, “Embedded pilot-aided channel estimation for OTFS in delay–doppler channels,” IEEE Transactions on Vehicular Technology, vol. 68, no. 5, pp. 4906–4917, 2019
[2] G. Acosta-Marum and M. A. Ingram, ‘‘Six time- and frequency-selective empirical channel models for vehicular wireless LANs,’’ IEEE Veh. Technol. Mag., vol. 2, no. 4, pp. 4–11, Dec. 2007.
[3] H. Shaiek, R. Zayani, Y. Medjahdi, and D. Roviras, “Analytical analysis of SER for beyond 5G post-OFDM waveforms in presence of high power amplifiers,” IEEE Access, vol. 7, pp. 29 441–29 452, 201.
