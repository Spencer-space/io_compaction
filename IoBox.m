function S = IoBox(h_hat,Pc,T_stop,plot_stuff,p)

clear S;
% Box model for Io,
% h_hat = non dimensionalised emplacement rate
% T_stop = homologous temperature below which to stop emplacement

% take qf = 0;
qf = 0;

% Dimensional Parameters
K_0 = 10^-7; % reference permeability (m^2)
rho = 3000; % density (kg/m^3)
del_rho = 500; % density difference (kg/m^3)
g = 1.5; % gravity (m/s^2)
L = 4e5; % latent heat (J/kg)
kappa = 1e-6; % Thermal diffusivity (m^2/s)
c = 1200; % specific heat (J/kg/K)
T_l = 1350; % relative melting point (above surface T) (K)
n = 3; % permeability exponent
eta_l = 1; % basalt melt viscosity (Pas)
eta = 1e20; % viscosity (Pas)

r_s = 1820e3; % Io radius (m)
r_m = 700e3/1820e3; % normalise core radius

Psi_ref = 1e14/(4/3 * pi *(r_s^3 - 700e3^3)); % reference tidal heating (W/m^3)
q_0 = Psi_ref*r_s/rho/L; % reference velocity (m/s)
phi_0 = (q_0*eta_l/(K_0*del_rho*g))^(1/n); % reference porosity

zeta_0 = eta/phi_0;
P_0 = zeta_0*q_0/r_s;
S.P_c = -Pc*P_0;

% Dimensionless Parameters
St = L/c/T_l; % Stefan number
Pe = q_0*r_s/kappa; % Peclet number
Psi = 1; % mantle heating rate
Psi_l = 1; % lithosphere heating rate

% Use shooting method to find lithosphere thickness, see functions for details
opts = odeset('reltol',1e-6);
[l,a,b,iter] = bisect(1e-4,1-r_m-1e-4,1e-6);
% Now with known r_l solve ODEs on correct domain
[x,y] = ode45(@odes1,[1-l 1],[Psi/3 *(1-l - r_m^3/(1-l)^2); 1; 0],opts);

if exist('p') == 0
    r = linspace(r_m,1,1000); % create position vector
else r = linspace(r_m,1,p.ni); % create position vector
end
dr = r(2)-r(1);
r1 = r(r>(1-l)); % Portion of position vector corresponding to lithosphere
r2 = r(r<(1-l)); % Portion of position vector corresponding to mantle

T = 1.*(r<=(1-l)); % Mantle is on the liquidus
T(r>(1-l)) = interp1(x,y(:,2),r1); % Interpolate the ODE temp solution onto lithosphere grid

qe = 0*r.^0; % qe is zero in mantle
qe(r>(1-l)) = interp1(x,y(:,1),r1); % Interpolate the ODE qe solution onto lithosphere grid

dT(r>(1-l)) = interp1(x,y(:,3),r1);

q = zeros(1,length(r));
phi = zeros(1,length(r));
Pbl = zeros(1,length(r));
P = zeros(1,length(r));

rl = 1-l; % radial position of lithosphere boundary
[phi_bl P_bl Z] = decompacting_boundary_layer(Pc,l);

delta = eta*q_0/(phi_0*del_rho*g*r_s^2);
q(r<rl) = Psi/3 .*(r2 - r_m^3./r2.^2);
phi(r<rl) = q(r<rl).^(1/n) + interp1(Z,phi_bl,(rl-r2)/delta) - (Psi/3 *(rl - r_m^3/rl^2))^(1/n); % use boundary layer formulation for q
u = -q-qe; % u from conservation of mass

P(r<rl) = -Psi./(q(r<rl).^(1/n)) + interp1(Z,P_bl,(rl-r2)/delta) + Psi./((Psi/3 *(rl - r_m^3/rl^2))^(1/n)); % outer solution + inner solution - P_inf (from BL calc)
P(isinf(P)) = 0;

P1 = -Psi./(q(r<rl).^(1/n));
P2 = interp1(Z,P_bl,(rl-r2)/delta);

% % ****************************************** %
% % No decompacting BL
% q(r<rl) = Psi/3 .*(r2 - r_m^3./r2.^2);
% phi(r<rl) = q(r<rl).^(1/n);
% u = -q-qe; % u from conservation of mass
% % ****************************************** %

% Store in outputting structure
S.r = r;
S.T = T;
S.qe = qe;
S.dT = dT;
S.q = q;
S.u = u;
S.phi = phi;
S.P = P;
S.M = h_hat*(1-T).*(qe>0).*(T>T_stop);

P = P*P_0/1e6;
T = T*T_l + 150; % Put temperature into dimensional units and shift to absolute T
u = u *Psi_ref*r_s/(rho*L); % dimensionalise velocity
q = q *Psi_ref*r_s/(rho*L); 
qe = qe *Psi_ref*r_s/(rho*L);
r = r * r_s; % dimsonsionalise radial position
rl = rl * r_s;

% Elastic thickness is the point where emplacement stops (defined by T_stop)
for i = 2:length(T)
    if T(i) < T_stop*(T_l+150) && T(i-1) > T_stop*(T_l+150)
        elas_thick = (r_s - r(i))/1000;
        S.re = elas_thick;
    end
end
% if rl hasn't been created set to zero
if ~isfield(S,'re')
    S.re = 0;
    elas_thick = 0;
end

if plot_stuff ~= 0
    ss_plot(T,phi,q*(100*60*60*24*365),qe*(100*60*60*24*365),u*(100*60*60*24*365),P,r);
    
    erupt_rate = u(end) *(100*60*60*24*365) % erupt rate in cm/yr
    cond_heat = -kappa*rho*c* dT(end)*T_l/r_s * (4*pi*r_s^2); % global conductive heat flux (W) kappa*rho*c = conductivity
    volc_heat = rho*qe(end) *(4*pi*r_s^2)*(c*T_l + L);
end

function [m,a,b,iter] = bisect(a,b,tol)
    % Looking for the value of lithosphere thickness that gives the correct surface temperature
    % Try two values and look for sign change, if there is one, narrow the range, if not then try another inteval.
    % Two values to try are 'a' and 'b'
    % When the difference between quesses is within a tollerance, success :-)
    iter = 0;
    fa = shoot(a); fb = shoot(b); % return surface temperatures and place them (hopefully) either side of zero.
    if fa*fb > 0
        %warning('bisect: Signs at ends of interval are the same');
        % Try looking for another interval
        for tmp = 0.1:.1:1
            c = a+tmp*(b-a);
            fc = shoot(c); 
            %if fa*fc < 0, b = c; fb = fc; warning(['bisect: Using reduced interval ',num2str([a b])]); break; end
            if fa*fc < 0, b = c; fb = fc; break; end
        end
        % Otherwise give up
        if tmp==1
            warning('bisect: Could not find an interval with a sign change');
            m = NaN;
            return;
        end
    end
    % Goes here when two guesses change sign (i.e. sit either side of surface temperature)
    while b-a>tol
        % Try halfway through the current interval, if it changes sign that's new interval, if not the leftover must be the inteval.
        iter = iter+1;
        m = (a+b)/2;
        fm = shoot(m);
        if fm*fa<0 
           b = m;
        else
           a = m;
           fa = fm;
        end
    end
end

function out = shoot(l)
    % solve ODEs for a guess of the lithosphere thickness, return the predicted surface temperature
    % Guesses for qe and dT take account of whether qe is zero or not. If it's zero in dT case, use q_e BC with q_e = 0.
    [x1,y1] = ode45(@odes1,[1-l 1],[Psi/3 *(1-l - r_m^3/(1-l)^2); 1; 0],opts);
    out = y1(end,2);
end

function dydx = odes1(x,y)
    % stiff ODE solver for the 3 coupled 1st order equations
    qe = y(1,:);
    T = y(2,:);
    dT = y(3,:);
    dydx(1,:) = - h_hat*(1-T).*(qe>0).*(T>T_stop) -2*qe/x; % M = 0 if qe<0
    dydx(2,:) = dT;
    dydx(3,:) = -Pe*qe*dT - Pe*St*Psi_l - 2*dT/x - Pe*h_hat*(1-T)*(St + (1-T)) .*(qe>0).*(T>T_stop); % Energy equation rewritten for d2T/dr2. M = 0 if qe<0
end

function lines = ss_plot(T,phi,q,qe,u,P,r)
    for ii=2:length(phi)
        if phi(ii)<=1e-5 && phi(ii-1)>1e-5
            lith = r(ii);
        end
    end
    
    figure('Units','centimeters','Position',[15 15 50 25],'PaperPositionMode','auto');
    fig1 = subplot(1,100,[1,30]);
    patch(fig1,[-10 -10 10 10],[lith/1000 r_s/1000 r_s/1000 lith/1000],[0.852 0.852 0.852])
    hold on
    patch(fig1,[-10 -10 10 10],[600 700 700 600],[0.632 0.5 0.5])
    gradient_patch = zeros(1,4,3);
    gradient_patch(1,1,:) = [0.696 0.696 0.696];
    gradient_patch(1,2,:) = [0.9 0.516 0.168];
    gradient_patch(1,3,:) = [0.9 0.516 0.168];
    gradient_patch(1,4,:) = [0.696 0.696 0.696];
    patch(fig1,[-10 -10 10 10],[700 lith/1000 lith/1000 700],gradient_patch)
    line = plot(fig1,q+qe,r/1000,'linewidth',2);
    line = plot(fig1,[-10 10],[1820-elas_thick 1820-elas_thick],'k--','linewidth',1.5);
    line = plot(fig1,u,r/1000,'linewidth',2,'color',[0 0.416 0.1]);
    line = plot(fig1,[-10 10],[700 700]);
    line = plot(fig1,[0 0], [700 1820],':','linewidth',2);
    
    axis([-10 10 600 1820]);
    set(fig1,'Units','normalized','FontUnits','points','FontSize',24,'FontName','Times', 'Layer', 'top')
    ylabel(fig1,{'Radial position $r$ (km)'},'FontUnits','points', 'interpreter','latex','FontSize', 24,'FontName','Times')
    xlabel(fig1,{'Upwelling flux (cm/yr)'},'FontUnits','points','interpreter','latex','FontSize', 24,'FontName','Times')
    title(fig1,'a)','position',[-10 1825], 'FontUnits','points','Fontweight','normal','FontSize',24,'FontName','Times')
    %legend(fig3,'Liquid','Solid','location','southwest','FontSize',14)
    
    fig2 = subplot(1,100,[40,70]);
    patch(fig2,[0 0 2000 2000],[lith/1000 r_s/1000 r_s/1000 lith/1000],[0.852 0.852 0.852])
    hold on
    patch(fig2,[0 0 2000 2000],[600 700 700 600],[0.632 0.5 0.5])
    patch(fig2,[0 0 2000 2000],[700 lith/1000 lith/1000 700],gradient_patch)
    line = plot(fig2,T,r/1000,'linewidth',2);
    line = plot(fig2,[0 2000],[700 700]);
    line = plot(fig2,[0 2000],[1820-elas_thick 1820-elas_thick],'k--','linewidth',1.5);
    axis([0 2000 600 1820]);
    set(fig2,'Units','normalized','FontUnits','points','FontSize',24,'FontName','Times', 'Layer', 'top')
    xlabel(fig2,{'Temperature (K)'},'FontUnits','points','interpreter','latex','FontSize', 24,'FontName','Times')
    title(fig2,'b)','position',[0 1825],'FontUnits','points','Fontweight','normal','FontSize',24,'FontName','Times')
    set(gca,'Yticklabel',[])

    fig3 = subplot(1,100,[76,86]);
    patch(fig3,[0 0 6 6],[lith/1000 r_s/1000 r_s/1000 lith/1000],[0.852 0.852 0.852])
    hold on
    patch(fig3,[0 0 6 6],[600 700 700 600],[0.632 0.5 0.5])
    patch(fig3,[0 0 6 6],[700 lith/1000 lith/1000 700],gradient_patch)
    line = plot(fig3,phi*phi_0*100,r/1000,'linewidth',2);
    line = plot(fig3,[0 6],[1820-elas_thick 1820-elas_thick],'k--','linewidth',1.5);
    line = plot(fig3,[0 6],[700 700]);
    axis([0 6 600 1820]);
    set(fig3,'Units','normalized','XTick',0:2:6,'FontUnits','points','FontSize',24,'FontName','Times', 'Layer', 'top')
    xlabel(fig3,{'Porosity (\%)'},'FontUnits','points','interpreter','latex','FontSize', 24,'FontName','Times')
    title(fig3,'c)','position',[0 1825], 'FontUnits','points','Fontweight','normal','FontSize',24,'FontName','Times')
    set(gca,'Yticklabel',[])
    
    fig4 = subplot(1,100,[90,100]);
    patch(fig4,[-100 -100 10 10],[lith/1000 r_s/1000 r_s/1000 lith/1000],[0.852 0.852 0.852])
    hold on
    patch(fig4,[-100 -100 10 10],[600 700 700 600],[0.632 0.5 0.5])
    patch(fig4,[-100 -100 10 10],[700 lith/1000 lith/1000 700],gradient_patch)
    line = plot(fig4,[0 0], [700 1820],'-k');
    line = plot(fig4,P(2:end),r(2:end)/1000,'linewidth',2);
    line = plot(fig4,[-100 10],[1820-elas_thick 1820-elas_thick],'k--','linewidth',1.5);
    line = plot(fig4,[-100 10],[700 700]);
    axis([-100 10 600 1820]);
    set(fig4,'Units','normalized','XTick',-100:50:0,'FontUnits','points','FontSize',24,'FontName','Times', 'Layer', 'top')
    xlabel(fig4,{'Pressure (MPa)'},'FontUnits','points','interpreter','latex','FontSize', 24,'FontName','Times')
    title(fig4,'d)','position',[-100 1825], 'FontUnits','points','Fontweight','normal','FontSize',24,'FontName','Times')
    set(gca,'Yticklabel',[])
    
    linkaxes([fig1 fig2 fig3 fig4],'y');
    
    AxesHandle=findobj(gcf,'Type','axes');
    set(AxesHandle(4),'Position',[0.07,0.09,0.25,0.86]);
    set(AxesHandle(3),'Position',[0.35,0.09,0.25,0.86]);
    set(AxesHandle(2),'Position',[0.63,0.09,0.15,0.86]);
    set(AxesHandle(1),'Position',[0.81,0.09,0.15,0.86]);
    
end

end