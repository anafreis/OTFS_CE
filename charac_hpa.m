function [s, K0theo, sigmad2theo] = charac_hpa(coeff_IBO_m1dB, inhpa, vout1, Np)
u=abs(inhpa);
% Estimation of polynomial complex coefficients
s = polyfit(u,(1./u).*abs(vout1).*exp(1i*(angle(vout1)-angle(inhpa))),Np);

% Calcul analytique de E (rho puissance n)
sigmai = sqrt(1/2)*coeff_IBO_m1dB;
E_rho_n = zeros(1,Np+1);
for k = 1:2:2*Np+2
    produit = 1;
    for kk = 1:((k-1)/2)+1
        produit = produit*(2*(kk-1)+1);
    end    
    E_rho_n(k) = sqrt(pi/2)*sigmai^k*produit;
end
for k = 2:2:2*Np+2
    E_rho_n(k) = (sqrt(2)*sigmai)^k*factorial(k/2);
end

% Calcul analytique de K0: Equation 10
K0theo = 0.5*2*s(Np+1);
for kk = 2:Np+1
    K0theo = K0theo+0.5*(kk+1)*E_rho_n(kk-1)*s(Np-kk+2);
end    

% calcul théorique de sigmad2 en utilisant K0theo
 AA = 0;
  for k1 = 1:Np+1
      for k2 = 1:Np+1
          AA = AA+s(Np-k1+2)*s(Np-k2+2)'*E_rho_n(k1+k2);
      end    
 end 
 sigmad2theo = real(AA-abs(K0theo)^2*E_rho_n(2));
end