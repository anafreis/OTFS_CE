function [s_AMAM_3GPP, s_AMPM_3GPP, val_m1dB_hpa] = function_HPA_WONG5_Poly(fig)
%   input fig: plot figures if fig == 1
%
%   output s_AMAM_3GPP: polynom coefficients of the AM/AM
%   curve(Volts/Volts), order 15.
%   output s_AMPM_3GPP: polynom coefficients of the AM/PM
%   curve(Volts/degrees), order 15.
%   output val_m1dB_HPA: input voltage corresponding to the compression
%   point at -1dB for HPA 3GPP
%
%   (1) HPA1 with AM/AM and AM/PM conversion given in document: R4-163314.docx
%   from 3GPP. This is an HPA model at frequency 4GHz. AM/AM and AM/PM 
%   curves are given in fig. 3 of the docx document. These conversions are 
%   valid from an input of -infinty dBm to 10dBm. The maximum output
%   power is 30dBm (1W). A polynomial fitting of AM/AM and AM/PM
%   conversions have been done with polynoms of order 9. The polynoms are
%   given below:
%   pam = [7.9726e-12  1.2771e-9  8.2526e-8  2.6615e-6  3.9727e-5  2.7715e-5  -7.1100e-3  -7.9183e-2  8.2921e-1  27.3535].';
%	ppm = [9.8591e-11  1.3544e-8  7.2970e-7  1.8757e-5  1.9730e-4  -7.5352e-4 -3.6477e-2  -2.7752e-1  -1.6672e-2  79.1553].';

%%   HPA1 
%
%   s_AMAM_3GPP are the polynom coefficients of the AM/AM curve
%   (Volts/Volts), order 15.
%
%   s_AMPM_3GPP are the polynom coefficients of the AM/PM curve
%   (Volts/degrees), order 15.
%
%   The two polynoms given in [1] give :
%   AM/AM conversion from dBm to dBm
%   AM/PM conversion from dBm to degrees
%========================================
%   (1) The first step is to compute AM/AM and AM/PM curve from Pin = -35dBm to +10dBm
%   (range where the polynom is valid) for 1000 values uniformly
%   distributed form -35 to +10.
pam = [7.9726e-12  1.2771e-9  8.2526e-8  2.6615e-6  3.9727e-5  2.7715e-5  -7.1100e-3  -7.9183e-2  8.2921e-1  27.3535].';
ppm = [9.8591e-11  1.3544e-8  7.2970e-7  1.8757e-5  1.9730e-4  -7.5352e-4  -3.6477e-2  -2.7752e-1  -1.6672e-2  79.1553].';
Pin = linspace(-35,10,1000);      % 1000 Values in dBm where AM/AM curve is defined
Pout = polyval(pam,Pin);
Phase_out = polyval(ppm,Pin);
if fig == 1
    figure(1)
    plot(Pin,Pout)
    title('AM/AM curve')
    xlabel('Input power in dBm')
    ylabel('Output power in dBm')
    grid
    figure(2)
    plot(Pin,Phase_out)
    title('AM/PM curve')
    xlabel('Input power in dBm')
    ylabel('Output angle in degrees')
    grid
end
%   (2) The second step is to compute AM/AM curve from Pin = -35dBm to +10dBm, 
%    but using volts instead of dBm for the x axis. Conversion from dBm to voltages is given by: 
%   (a) compute Pin in linear scale: Pin_linear=1e-3*10.^(Pin_dBm/10)
%   (b) compute the voltage, Vin_lin corresponding to Pin_lin in a 50 Ohms
%   load: Vin_lin = sqrt(Pin_lin*50)
%   The range [-35dBm +10dBm] corresponds to [0.004V 0.7V]. Outside this voltage range, 
%   the polynomial expression is no more valid.
Pin_lin = 1e-3*10.^(Pin/10);
Vin_lin = sqrt(Pin_lin*50);
Pout_lin = 1e-3*10.^(Pout/10);
Vout_lin = sqrt(Pout_lin*50);
if fig == 1
    figure(3)
    plot(Vin_lin,Vout_lin)
    title('AM/AM curve')
    xlabel('Input signal in Volts')
    ylabel('Output signal in Volts')
    grid
end
%   (3) When plotting AM/AM curve using volts we can see (figure 3) that 
%   the curve from v = 0 to v = 0.2 is different from a straight line. 
%   It is necessary to have a constant gain for low voltages. From v=0 to
%   v=0.2 we will replace the original AM/AM curve by a straight line. 
%   A new "modified" AM/AM conversion curve is then obtained. 
vv = 0.2;
tt = abs(Vin_lin-vv);
[x1, y1] = min(tt);
pente = Vout_lin(y1)/Vin_lin(y1);
Vout_lin_modified(Vin_lin > vv) = Vout_lin(Vin_lin > vv);
Vout_lin_modified(Vin_lin < vv) = pente*Vin_lin(Vin_lin < vv);
if fig == 1
    figure(4)
    plot(Vin_lin,Vout_lin, 'b')
    hold on
    plot(Vin_lin,Vout_lin_modified, 'r')
    title('AM/AM curve')
    legend('Original AM/AM curve','Modified AM/AM curve');
    xlabel('Input signal in Volts')
    ylabel('Output signal in Volts')
    grid
end
%   (4) The forth step consists in finding a polynomial fitting for the
%   "modified" AM/AM conversion with a polynom of order equal to 15.
s_AMAM_3GPP = polyfit(Vin_lin,Vout_lin_modified,15);
Vout_lin_modified_poly = polyval(s_AMAM_3GPP,Vin_lin);
if fig == 1
    figure(5)
    plot(Vin_lin,Vout_lin_modified, 'b')
    hold on
    plot(Vin_lin,Vout_lin_modified_poly, 'r')
    title('AM/AM curve')
    legend('Modified AM/AM curve','Polynomial AM/AM curve');
    xlabel('Input signal in Volts')
    ylabel('Output signal in Volts')
    grid
end
%   (5) The fith step consists in doing the conversion from dBm to Volts
%   for the AM/PM conversion and to fit with a polynom of order equal to
%   15.
s_AMPM_3GPP = polyfit(Vin_lin,Phase_out,15);
Phase_out_poly = polyval(s_AMPM_3GPP, Vin_lin);
if fig == 1
    figure(6)
    plot(Vin_lin,Phase_out, 'b')
    hold on
    plot(Vin_lin,Phase_out_poly, 'r')
    title('AM/PM curve')
    legend('Original AM/PM curve','Polynomial AM/PM curve');
    xlabel('Input signal in Volts')
    ylabel('Output signal in degrees')
    grid
end
%   (6) The last step is to compute the input voltage value corresponding
%   to the compression point àf -1dB. This voltage is called val_m1dB_HPA1
inhpa = linspace(0.05, max(Vin_lin), 1000);
outhpa = polyval(s_AMAM_3GPP, inhpa); 
outlin = pente*inhpa;
tt = abs(outlin.^2*(10^-0.1)-outhpa.^2);
[x1, y1] = min(tt);
val_m1dB_hpa = inhpa(y1);
inhpa = linspace(0, 0.7, 10000);
outhpa1 = polyval(s_AMAM_3GPP, inhpa);  
phasehpa = polyval(s_AMPM_3GPP, inhpa); 
if fig == 1
    figure(7)
    plot(inhpa,outhpa1)
    hold on
    plot(inhpa,pente*inhpa,'r')
    plot(val_m1dB_hpa*ones(1,10), linspace(0,0.5*pente,10),'-m')
    grid
    xlabel('Input signal in Volts')
    ylabel('Output signal in Volts')
    legend('AM/AM conversion with s-AMAM-3GPP polynom','Linear AM/AM conversion','-1dB saturation point 3GPP HPA')
    title('Final AM/AM curve with -1dB compression point 3GPP HPA ')
    figure(8)
    plot(inhpa, phasehpa,'b')
    grid
    xlabel('Input signal in Volts')
    ylabel('Output signal in degrees')
    title('AM/PM conversion with s-AMPM-3GPP polynom')
end
