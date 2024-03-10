% Parameters
a = 0.3;
b = 0.4;
% Initial conditions - First set
n = 50;
x1 = zeros(1, n);
x1(1) = 0;
x1(2) = 0;
% Henon map iterations - First set
for i = 3:n
x1(i) = 1 - a * x1(i-1)^2 + b * x1(i-2);
end
% Initial conditions - Second set
x2 = zeros(1, n);
x2(1) = 0.00001;
x2(2) = 0.00001;
% Henon map iterations - Second set
for i = 3:n
x2(i) = 1 - a * x2(i-1)^2 + b * x2(i-2);
end
% Plotting both sets together
figure;
plot(1:n, x1, 'b', 1:n, x2, 'r');
title('Comparison of 1D Henon Maps for Different Initial Conditions');
xlabel('n');
ylabel('x_n');
legend('x(1) = 0.1', 'x(1) = 0.0001');
axis ("auto[x]");

%octave doesnt have xline function so we implement it
xline = @(xval, varargin) line([xval xval], ylim, varargin{:});
a_values = linspace(-0.2, 1.7, 100000);
b = 0;
x = [0, 0]; % Initializing x with 0, 0

for i = 3:length(a_values)
x(i) = 1 - a_values(i) * x(i - 1)^2 + b * x(i - 2);
end
plot(a_values, x, 'o', 'MarkerSize', 1);
grid on;
xlabel('Parameter a');
ylabel('X value');
title('Hénon Map Bifurcation Diagram for b = 0 and a in range (-0.2, 1.7)');
xline(1.63);
xline(1.48);
% Setting x-axis ticks for every 0.1
xticks(-1:0.1:2)


