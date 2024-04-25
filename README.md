# OTFS_CE
This project presents codes for the LS-LSTM-NN channel estimation affected by HPA impairments proposed for OTFS channels in "LSTM-Based Time-Frequency Domain Channel Estimation for OTFS Modulation".

This project presents:
	+ The SFFT-based OTFS Tx-Rx implementation with the TCE [1] channel estimation
		---- Communication scenario provided in Matlab
	+ The full training and testing phases LS-LSTM-NN channel estimators proposed
		---- LSTM network implementation provided in Python
----------------------------------------------------------------------------------------------------------------------
The project's folder is structured as follows:
	+ Matlab_Codes/main_OTFS_CE_NLD.m: Present the main simulation file. Here the user needs to define the simulation parameters (Speed, channel model, modulation order, HPA IBO, [...]). This file will save the datasets including the LS estimation and TCE estimation.
	+ Matlab_Codes/Channel_functions.m: Includes the pre-defined vehicular channel models [2] for different mobility conditions.
	+ Matlab_Codes/OTFS_functions.m: Includes functions for SFFT-based OTFS transmission.
	+ Matlab_Codes/Datasets_Generation_NLD: The testing and training datasets are generated in the path Python_Codes - NLD
	+ Python_Codes/LSTM_Training.py:  The LSTM-NN training is performed by employing the training dataset. 500 models are saved and considered as average (in order to obtain reproducible results) in the next step.
	+ Python_codes/LSTM_Avg_Model: The latest 50 trained models are averaged.	  
	+ Python_Codes/LSTM_Testing.py: The LSTM-NN model is tested in considering the testing datasets and the results are saved in .mat files.
	+ Matlab_Codes/Results_Processing_NLD and Matlab_Codes/Plotting_Results_[...]: Results from python are processed, saved and ploted for the different metrics

Additional files:
	+ Matlab_codes/[NLD;hpa_saleh;function_HPA_WONG5_Poly;charac_hpa;]: Are related to the Memoryless Polynomial HPA described in [3].
	+ Matlab_codes/circulant: Is used to obtain the circulant matrix in the TCE method.
----------------------------------------------------------------------------------------------------------------------
[1] P. Raviteja, K. T. Phan, and Y. Hong, “Embedded pilot-aided channel estimation for OTFS in delay–doppler channels,” IEEE Transactions on Vehicular Technology, vol. 68, no. 5, pp. 4906–4917, 2019
[2] G. Acosta-Marum and M. A. Ingram, ‘‘Six time- and frequency-selective empirical channel models for vehicular wireless LANs,’’ IEEE Veh. Technol. Mag., vol. 2, no. 4, pp. 4–11, Dec. 2007.
[3] H. Shaiek, R. Zayani, Y. Medjahdi, and D. Roviras, “Analytical analysis of SER for beyond 5G post-OFDM waveforms in presence of high power amplifiers,” IEEE Access, vol. 7, pp. 29 441–29 452, 201.
