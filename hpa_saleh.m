function [vout K0est sigmaD2est]  = hpa_saleh(vin,alpha1,beta1,alpha2,beta2,vsat)
    a0=abs(vin);
    a02=a0.^2;
    theta=angle(vin);
    A=(alpha1.*a0)./(1.+(beta1.*(a0.^2)));
    B=(alpha2.*(a0.^2))./(1.+(beta2.*(a0.^2)));
    recu1 = A.*cos(theta + B);
    recu2 = A.*sin(theta + B);
    Send113 = complex(recu1,recu2);
    vout = Send113;
    VoutPh =angle(vout);
    
  K0est=vsat^2*mean(exp(j*B).*(vsat+a02).^-1)-vsat^2*mean(exp(j*B).*a02.*(vsat+a02).^-2);
  K0est=K0est+j*vsat^2*alpha2*mean(exp(j*B).*(vsat+a02).^-2.*a02)-j*vsat^2*alpha2*mean(exp(j*B).*(vsat+a02).^-3.*(a02.^2));
  sigmaD2est=mean(abs(vout).^2)-abs(K0est)^2*mean(a02);

end
