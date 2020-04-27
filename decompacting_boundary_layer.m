function [phi P Z] = decompacting_boundary_layer(Pc,l)
% 14 May 2019
%Pc is overpressure condition
%l is non-dimensional lithosphere thickness

if Pc == 0
    Pc = 1e-8;
end

n = 3;
Psi = 1;
opts = odeset('reltol',1e-6,'abstol',1e-6);

c = 1200; % (J/kg/K) specific heat capacity
L = 4e5; % (J/kg) latent heat
rho = 3000; % (kg/m3) density
eta = 1e20; % (Pas) solid viscosity
eta_l = 1; % (Pas) liquid viscosity
K_0 = 1e-7; % (m2) reference rigidity
R = 1820e3; % (m) radius of Io
g = 1.5; % (m/s2) gravity
del_rho = 500; % density difference

psi_0 = 1e14/(4/3 * pi *(R^3 - 700e3^3)); % mantle tidal heating (W/m3)
q_0 = psi_0*R/rho/L;
phi_0 = (q_0*eta_l/(K_0*del_rho*g))^(1/n);
zeta_0 = eta/phi_0;
P_0 = zeta_0*q_0/R;
delta_0 = zeta_0*q_0/(del_rho*g*R^2);

r_m = 700/1820; % non-dimensional base of Io's mantle

qinf = 1/3 *((1-l)-r_m^3/(1-l)^2); % calculate analytical velocity at base of lithosphere
%qinf = 1;

phiinf = qinf^(1/n);

epshat = phi_0/delta_0;
%epshat = 1;

% calculate base of the extraction region
Ze = epshat*qinf/Pc *log(qinf^(1/n)*Pc/Psi +1);

[Z,y] = ode45(@odes,[Ze (1-l)/delta_0],[phiinf; Pc],opts);
phi = y(:,1);
P = y(:,2);

% calculate solution in extraction region
Z_ex = linspace(0,Ze,100)';
phi_ex = Psi/Pc *(exp(Pc*Z_ex/(epshat*qinf))-1);
P_ex = Pc*ones(100,1);

% append extraction region solution to rest of solution (don't overlap point)
Z = [Z_ex; Z(2:end)];
phi = [phi_ex; phi(2:end)];
P = [P_ex; P(2:end)];

%[pks,locs,w,p] = findpeaks(phi,Z,'WidthReference','halfprom');
% 
% phi_max = pks(1);
% peak_thick = w(1);

return
    
figure(1); clf; set(gcf,'units','centimeters','paperpositionmode','auto','position',[15 15 30 20]);
set(gcf,'DefaultAxesFontSize',12,'DefaultTextFontSize',12);
cols = get(0,'DefaultAxesColorOrder');
width = 1.5;
yax = [0 10];
ax1(1) = subplot(1,2,1);
    plot(phi,Z,'-','color',cols(1,:),'linewidth',width);
    ylim(yax);
    xlim([0 4.5]);
    set(gca,'Units','normalized','FontUnits','points','FontSize',20,'FontName','Times','YDir','reverse')
    xlabel('$\phi$','interpreter','latex');
    ylabel('Z');
    grid on;
    ax1(2) = subplot(1,2,2);
    plot(P,Z,'-','color',cols(1,:),'linewidth',width);
    ylim(yax);
    xlim([-1 2]);
    set(gca,'Units','normalized','FontUnits','points','FontSize',20,'FontName','Times','YDir','reverse')
    xlabel('$P$','interpreter','latex');
    grid on;
    
%   return
    
% phase plane
figure(2); clf; set(gcf,'units','centimeters','paperpositionmode','auto','position',[2 2 12 8]);
set(gcf,'DefaultAxesFontSize',12,'DefaultTextFontSize',12);
width = 1.5;
    pphi = linspace(0,4.5,100);
    PP = linspace(-2,2,100);
    [pphi,PP] = meshgrid(pphi,PP);
    y = [reshape(pphi,1,[]); reshape(PP,1,[]); reshape(ones(size(pphi,1),size(pphi,2)),1,[])];
    dydx = odes(0,y);
    dphidZ = reshape(dydx(1,:),size(pphi,1),size(pphi,2));
    dPdZ = reshape(dydx(2,:),size(pphi,1),size(pphi,2));
    contour(pphi,PP,dphidZ,[0 0],'k','linewidth',width);
    hold on;
    contour(pphi,PP,dPdZ,[0 0],'k','linewidth',width);
    xlabel('\phi');
    ylabel('P');
    axis([0 4.5 -1.5 2]);
    %title('Phase plane solution', 'FontUnits','points','Fontweight','normal','FontSize', 24,'FontName','Times')
    xlabel('$\phi$','interpreter','latex');
    ylabel('$P$','interpreter','latex');
    set(gca,'Units','normalized','FontUnits','points','FontSize',20,'FontName','Times')
     
    skip = 5;
    u = dphidZ; v = dPdZ; u = u./(u.^2+v.^2).^(1/2); v = v./(u.^2+v.^2).^(1/2);
    quiver(pphi(skip:skip:end,skip:skip:end),PP(skip:skip:end,skip:skip:end),u(skip:skip:end,skip:skip:end),v(skip:skip:end,skip:skip:end),'color',0.8*[1 1 1]);
    
    hold on; plot(phi,P,'-','color',cols(1,:),'linewidth',width);
    
% re-solve
Pc_range = [-9e6/P_0,-6e6/P_0,-3e6/P_0,-0e6/P_0];
% 
% Pc_range = linspace(-2,0,4);
% Pc_range*P_0
% for i=1:4
%     Pc = Pc_range(i);
%     [Z,y] = ode45(@odes,[0 20],[phiinf; Pc],opts);
%     phi = y(:,1);
%     P = y(:,2);
%     
%     Pc_name = sprintf('%i',round(Pc,2));
%     Pc_name = sprintf('%i',round(Pc,2));
%     % add new solutions to plot
%     figure(1);
%     axes(ax1(1)); hold on; plot(phi*phi_0*100,Z*delta_0*R/1000,'-','linewidth',width,'DisplayName',Pc_name);
% %     legend('-DynamicLegend');
% %     legend('show');
%     axes(ax1(2)); hold on; plot(P*P_0/1e6,Z*delta_0*R/1000,'-','linewidth',width,'DisplayName',Pc_name);
% %     legend('-DynamicLegend');
% %     legend('show');
% end
% legend(ax1(1),'P_{c} = -1.5','P_{c} = -1.0','P_{c} = -0.5','P_{c} = 0.0','location','southeast');
% legend(ax1(2),'P_{c} = -1.5','P_{c} = -1.0','P_{c} = -0.5','P_{c} = 0.0','location','southwest');
% title(ax1(1),'Boundary layer porosity', 'FontUnits','points','Fontweight','normal','FontSize', 24,'FontName','Times')
% title(ax1(2),'Boundary layer pressure', 'FontUnits','points','Fontweight','normal','FontSize', 24,'FontName','Times')

%print(gcf,'-depsc2',[fn,'_fig1'],'-loose'); %fixPSlinestyle([fn,'_fig1']);

% % example of solution with finite extraction region
% Ze = (-What)/(-Pc)*log(qinf^(1/n)*(-Pc)/Psi+1);
% Z2 = linspace(0,Ze,100);
% phi2 = Psi/(-Pc)*(exp((-Pc)/(-What)*Z2)-1);
% q2 = phi2.^n;
% P2 = Pc*Z2.^0;
% 
% figure(3); clf; set(gcf,'units','centimeters','paperpositionmode','auto','position',[2 2 16 8]);
% set(gcf,'DefaultAxesFontSize',12,'DefaultTextFontSize',12);
% cols = get(0,'DefaultAxesColorOrder');
% width = 1.5;
% yax = [0 10];
% ax1(1) = subplot(1,3,1);
%     plot(phi,Z+Ze,'-','color',cols(2,:),'linewidth',width);
%     hold on; plot(phi2,Z2,'--','color',cols(2,:),'linewidth',width);
%     ylim(yax);
%     set(gca,'YDir','reverse');
%     xlabel('$\phi$','interpreter','latex');
%     ylabel('Z');
% ax1(2) = subplot(1,3,2);
%     plot(P,Z+Ze,'-','color',cols(2,:),'linewidth',width);
%     hold on; plot(P2,Z2,'--','color',cols(2,:),'linewidth',width);
%     ylim(yax);
%     set(gca,'YDir','reverse');
%     xlabel('$P$','interpreter','latex');
% ax1(3) = subplot(1,3,3);
%     plot(q,Z+Ze,'-','color',cols(2,:),'linewidth',width);
%     hold on; plot(q2,Z2,'--','color',cols(2,:),'linewidth',width);
%     ylim(yax);
%     set(gca,'YDir','reverse');
%     xlabel('$q$','interpreter','latex');

%print(gcf,'-depsc2',[fn,'_fig3'],'-loose'); %fixPSlinestyle([fn,'_fig3']);

% % add to phase plane
% figure(2);
%     plot(phi,P,'-','color',cols(2,:),'linewidth',width);
%     plot(phi2,P2,'--','color',cols(2,:),'linewidth',width);

%print(gcf,'-depsc2',[fn,'_fig2'],'-loose'); %fixPSlinestyle([fn,'_fig2']);



%% subfunctions
function dydx = odes(x,y)
% d/dZ [phi; P]
      dydx = [ (Psi+y(1,:).*y(2,:))./(epshat*qinf); -(1-qinf./y(1,:).^n)];
end

end