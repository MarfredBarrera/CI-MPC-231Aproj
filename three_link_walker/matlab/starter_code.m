% ME193B/293B Feedback Control of Legged Robots
% UC Berkeley

clear
%% Define symbolic variables for cofiguration variables and mechanical parameters

syms q1 q2 q3 x y real
syms dq1 dq2 dq3 dx dy real
syms u1 u2 real

% Position Variable vector
q = [x;y;q1;q2;q3];

% Velocity variable vector
dq = [dx;dy;dq1;dq2;dq3];

% State vector
s = [q;dq];

% Inputs
u = [u1;u2];

% parameters
          
lL = 1;
lT = 0.5;
mL = 5;
mT = 10;
JL = 0;
JT = 0;
mH = 15;
g = 9.81;

% # of Degrees of freedom
NDof = length(q);


%% Problem 1: Lagrangian Dynamics
%Find the CoM position of each link

% Torso
pComTorso = [x + lT*sin(q3);...
             y + lT*cos(q3)];

% Leg 1
pComLeg1 = [x - lL/2*cos(deg2rad(270) - (q1 + q3));...
            y - lL/2*sin(deg2rad(270) - (q1 + q3))];

% Leg 2
pComLeg2 = [x + lL/2*cos(q2 + q3 - deg2rad(90));...
            y - lL/2*sin(q2 + q3 - deg2rad(90))];


% Leg 1
pLeg1 = [x - lL*cos(deg2rad(270) - (q1 + q3));...
            y - lL*sin(deg2rad(270) - (q1 + q3))];

% Leg 2
pLeg2 = [x + lL*cos(q2 + q3 - deg2rad(90));...
            y - lL*sin(q2 + q3 - deg2rad(90))];

% Find the CoM velocity of each link

% Torso
dpComTorso = simplify(jacobian(pComTorso, q)*dq);

% Leg 1
dpComLeg1 = simplify(jacobian(pComLeg1, q)*dq);

% Leg 2
dpComLeg2 = simplify(jacobian(pComLeg2, q)*dq);


%% Find absolute angular velocity associated with each link:
% Torso
dq3Absolute = dq3;
% Leg 1
dq1Absolute = dq3 + dq1;
% Leg 2
dq2Absolute = dq3 + dq2;

% Total Kinetic energy = Sum of kinetic energy of each link

% Torso
KETorso = 0.5*mT*dpComTorso(1)^2 + 0.5*mT*dpComTorso(2)^2 + 0.5*JT*dq3Absolute^2;

% Leg 1
KELeg1 = 0.5*mL*dpComLeg1(1)^2 + 0.5*mL*dpComLeg1(2)^2 + 0.5*JL*dq1Absolute^2;

% Leg 2
KELeg2 = 0.5*mL*dpComLeg2(1)^2 + 0.5*mL*dpComLeg2(2)^2 + 0.5*JL*dq2Absolute^2;

KEHip = 0.5*mH*dx^2 + 0.5*mH*dy^2;
% Total KE
KE = simplify(KETorso + KELeg1 + KELeg2 + KEHip);

% Total potential energy = Sum of Potential energy of each link

% Torso
PETorso = mT*g*pComTorso(2);

%Leg 1
PELeg1 = mL*g*pComLeg1(2);

% Leg 2
PELeg2 = mL*g*pComLeg2(2);

% Hip
PEHip = mH*g*y;
% Total PE
PE = simplify(PETorso + PELeg1 + PELeg2 + PEHip);

% Lagrangian

L = KE - PE;

% Equations of Motion
EOM = jacobian(jacobian(L,dq), q)*dq - jacobian(L, q)' ;
EOM = simplify(EOM);

% Find the D, C, G, and B matrices

% Actuated variables
qActuated = [q1;q2];

% D, C, G, and B matrices
[D, C, G, B] = LagrangianDynamics(KE, PE, q, dq, qActuated);


%%  Dynamics of Systems with Constraints
%Compute the Ground reaction Forces

% Compute the position of the stance foot (Leg 1) 
pst = [x - lL*cos(deg2rad(270) - (q1 + q3));...
       y - lL*sin(deg2rad(270) - (q1 + q3))];


% Compute the jacobian of the stance foot
JSt = jacobian(pst, q);


% Compute the time derivative of the Jacobian
dJSt = sym(zeros(size(JSt)));
for i = 1:size(JSt, 1)
    for j = 1:size(JSt, 2)
        dJSt(i, j) = simplify(jacobian(JSt(i, j), q)*dq);
    end
end

H = C*dq + G;
alpha = 0;
% Constraint Force to enforce the holonomic constraint:
FSt = - pinv(JSt*(D\JSt'))*(JSt*(D\(-H + B*u)) + dJSt*dq + 2*alpha*JSt*dq + alpha^2*pst);
FSt = simplify(FSt);

% Split FSt into 2 components: 
%   1. which depends on u and 
%   2. which does not depend on u 
% Note: FSt is linear in u

Fst_u = jacobian(FSt, u); % FSt = Fst_u*u + (Fst - Fst_u*u)
Fst_nu = FSt - Fst_u*u; % Fst_nu = (Fst - Fst_u*u)

%% Impact Map

% Compute the swing leg position (leg 2)
pSw = [x + lL*cos(q2 + q3 - deg2rad(90));...
       y - lL*sin(q2 + q3 - deg2rad(90))];

JSw = jacobian(pSw, q);

% postImpact = [qPlus;F_impact];
% Here, q, dq represent the pre-impact positions and velocities
[postImpact] = ([D, -JSw';JSw, zeros(2)])\[D*dq;zeros(2,1)];

% Post Impact velocities
dqPlus = simplify(postImpact(1:NDof));

% Impact Force Magnitude
Fimpact = simplify(postImpact(NDof+1:NDof+2));


%% Other functions

% swing foot velocity
dpSw = JSw*dq;

%% Export functions
if ~exist('./gen')
    mkdir('./gen')
end
addpath('./gen')

matlabFunction(FSt, 'File', 'gen/Fst_gen', 'Vars', {s, u});
matlabFunction(dqPlus, 'File', 'gen/dqPlus_gen', 'Vars', {s});
matlabFunction(pSw, 'File', 'gen/pSw_gen', 'Vars', {s});
matlabFunction(dpSw, 'File', 'gen/dpSw_gen', 'Vars', {s});
matlabFunction(pst, 'File', 'gen/pSt_gen', 'Vars', {s});
matlabFunction(pComLeg1, 'File', 'gen/pComLeg1_gen', 'Vars', {s});
matlabFunction(pComLeg2, 'File', 'gen/pComLeg2_gen', 'Vars', {s});
matlabFunction(pComTorso, 'File', 'gen/pComTorso_gen', 'Vars', {s});
matlabFunction(pLeg1, 'File', 'gen/pLeg1_gen', 'Vars', {s});
matlabFunction(pLeg2, 'File', 'gen/pLeg2_gen', 'Vars', {s});




%% [Part 1a] Compute the f and g vectors
% Fill up this
f = [dq;
    -inv(D)*(C*dq + G - JSt'*Fst_nu)];

g = [zeros(5,2);
    inv(D)*(B + JSt'*Fst_u)];

f = simplify(f);
g = simplify(g);

matlabFunction(f, 'File', 'gen/f_gen', 'Vars', {s});
matlabFunction(g, 'File', 'gen/g_gen', 'Vars', {s});

%% Change of Coordinates
% Transformation matrix:
T = [1 0 0 0 0;
     0 1 0 0 0;
     0 0 1 0 1;
     0 0 0 1 1;
     0 0 0 0 1];
d = [0;
     0;
     -pi;
     -pi;
     0];

%% [Part 1b] Output dynamics
% Fill up this
h = [0, 0, 0, 0, 1;
    0, 0, 1, 1, 0];

output_y = h*(T*q+d) - [pi/6; 0];

output_y = simplify(output_y);
matlabFunction(output_y, 'File', 'gen/output_y_gen', 'Vars', {s});

%% [Part 1c] Lie Derivatives
% Fill up this

% Lf_y = jacobian(output_y, q) * dq;
% Lg_y = zeros(2, 2);
% 
% Lf2_y = [jacobian(Lf_y, q), jacobian(output_y, q)] * f;
% LgLf_y = jacobian(output_y, q) * inv(D) * (B + JSt.'*Fst_u);

Lf_y = jacobian(output_y, s) * f;
Lg_y = jacobian(output_y, s) * g;

Lf2_y = jacobian(Lf_y, s) * f;
LgLf_y = jacobian(Lf_y, s) * g;

Lf_y = simplify(Lf_y);
Lf2_y = simplify(Lf2_y);
LgLf_y = simplify(LgLf_y);

matlabFunction(Lf_y, 'File', 'gen/Lf_y_gen', 'Vars', {s});
matlabFunction(Lf2_y, 'File', 'gen/Lf2_y_gen', 'Vars', {s});
matlabFunction(LgLf_y, 'File', 'gen/LgLf_y_gen', 'Vars', {s});

%% [Part 1d] Relabelling Matrix
% Fill up this

R = [1 0 0 0 0
    0 1 0 0 0
    0 0 0 1 0
    0 0 1 0 0
    0 0 0 0 1];

%% Problem 2
% ----helpers----
function output = phi(x1, x2)
    a = 0.9;

    output = x1 + 1/(2-a) * sign(x2)*abs(x2)^(2-a);
end

function output = psi(x1, x2)
    a = 0.9;

    output = -sign(x2)*abs(x2)^a - sign(phi(x1, x2))*abs(phi(x1, x2))^(a/(2-a));
end

function output = v(y1, y2, dy1, dy2)
    epsilon = 0.1;

    output = [1/epsilon^2 * psi(y1, epsilon*dy1);
        1/epsilon^2 * psi(y2, epsilon*dy2)];
end

function ctl = control(s)
    LgLf = LgLf_y_gen(s);
    Lf2 = Lf2_y_gen(s);
    y = y_gen(s);
    dy = Lf_y_gen(s);
    input_v = v(y(1), y(2), dy(1), dy(2));

    ctl = inv(LgLf) * (-Lf2 + input_v);
end

function ds = dynamics(t, s)
    f = f_gen(s);
    g = g_gen(s);

    u = control(s);

    ds = f + g * u;
end

function sPlus = impact(s)
    R = [1 0 0 0 0
        0 1 0 0 0
        0 0 0 1 0
        0 0 1 0 0
        0 0 0 0 1];
    qMinus = s(1:5);
    qPlus = R*qMinus;
    dqPlus = R*dqPlus_gen(s);
    sPlus = [qPlus; dqPlus];
end

function [value, isterminal, direction] = three_link_event(t, x)
    value      = x(3) + x(5) - pi - pi/8;  
    isterminal = 1;                   
    direction  = +1;                  
end


%% Simulate
x0 = [-0.3827; 0.9239; 2.2253; 3.0107; 0.5236; ...
      0.8653; 0.3584; -1.0957; -2.3078; 2.0323];
N = 10;

% Simulate
is_impact = false;
n_steps = 0;
t_cur = 0; t_f = 100;
x_cur = x0;
opt = odeset('Events', @three_link_event);
t = []; x = []; t_I = [];
u_log = []; Fst_log = [];

while (n_steps < N) | (t_cur ~= t_f) % take 10 steps
    if is_impact
        x_plus = impact(x_cur');
        x_cur = x_plus;
        n_steps = n_steps + 1;
        disp("taken step at"); disp(x_cur)
    
    else
        [tseg, xseg] = ode45(@dynamics, [t_cur, t_f], x_cur, opt);
        t = [t; tseg];
        x = [x; xseg];

        useg = zeros(length(tseg), 2);
        Fstseg = zeros(length(tseg), 2);

        for k = 1:length(tseg)
            useg(k, :) = control(xseg(k, :)');
            Fstseg(k, :) = Fst_gen(xseg(k, :)', useg(k, :)');
        end

        u_log = [u_log; useg];
        Fst_log = [Fst_log; Fstseg];

        if t_I
            t_I = [t_I; t_I(end)+length(tseg)];
        else
            t_I = [length(tseg)];
        end
        x_cur = xseg(end, :);
        t_cur = tseg(end, :);
    end
    is_impact = ~is_impact;
end

animateThreeLink(t, x(:, 1:5))

%% Plots
close all;

figure(1);
plot(x(:, 3)+x(:, 5)-pi, x(:,8)+x(:, 10))
xlabel("\theta_1");
ylabel("$\dot{\theta}_1$", 'Interpreter', 'latex');

figure(2);
hold on
plot(t, u_log(:, 1));
plot(t, u_log(:, 2));
hold off
xlabel('t')
ylabel('u')
legend(["u_1", "u_2"]);

figure(3);
hold on;
plot(t, Fst_log(:, 1));
plot(t, Fst_log(:, 2));
hold off;
xlabel('t')
ylabel('F_{st}')
legend(["F_{st}^1", "F_{st}^2"]);


figure(4);
plot(abs(Fst_log(:, 1) ./ Fst_log(:, 2)));
title("$|F_{st}^1 / F_{st}^2|$", 'Interpreter', 'latex')