function [output_HPA] = NLD(s_AMAM_3GPP,s_AMPM_3GPP,val_m1dB_hpa,IBO,s_OTFS)
 %% NL HPA: Polynomial Model
% Before amplifying the signal by the HPA it is necessary to scale
% it by the coefficient alpha (Equation 2) 
coeff_IBO_m1dB = val_m1dB_hpa*sqrt(10^(-IBO/10));
s_OTFS_HPA = coeff_IBO_m1dB*s_OTFS;
% The conversion function from the Polynomial model: Document R4-163314
output_HPA = polyval(s_AMAM_3GPP,abs(s_OTFS_HPA)).*exp(1i*angle(s_OTFS_HPA)+ 1i*polyval(s_AMPM_3GPP,abs(s_OTFS_HPA))*2*pi/360);

% Estimation of K0 and SIGMAD2: modeling the non-linear distortions
inhpa = linspace(0.001, 0.7, 10000);
vout1 = polyval(s_AMAM_3GPP,abs(inhpa)).*exp(1i*angle(inhpa)+ 1i*polyval(s_AMPM_3GPP,abs(inhpa))*2*pi/360);
% Computation of polynomial model of the HPA with order Np
%   Theoretical computation of K0 using the polynomial model: K0theo
%   Theoretical computation of Sigmad2 = sigmad2theo
Nphpa = 7; 
[~, K0theo, ~] = charac_hpa(coeff_IBO_m1dB, inhpa, vout1, Nphpa);

% Phase correction due to K0
input_HPA = reshape(s_OTFS,1,size(s_OTFS,1)*size(s_OTFS,2));
output_HPA_reshape = reshape(output_HPA,1,size(output_HPA,1)*size(output_HPA,2));    
output_HPA = exp(-1i*angle(K0theo))*sqrt(var(input_HPA))*output_HPA/sqrt(var(output_HPA_reshape)); 
end

