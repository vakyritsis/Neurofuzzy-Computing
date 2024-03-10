% A.
[w1, w2] = meshgrid(-4:0.1:4);
f = 1 + 3*w1 +w2 +2.5*(w1.^2 + w2.^2)
contour(w1, w2, f)
title("Contour lines")

% B.
x = linspace(-5, 5);
y = -3 * x;
hold on
axis([-5 5 -5 5])
plot(x, y)
plot(1, 2, "o")
plot(-2, 1, "o")
text(1, 2, 'p1', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right')
text(-2, 1, 'p2', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right')
title("Optimal decision boundary")
hold of

% C.
% Generating grid for contour plot
[w1, w2] = meshgrid(-4:0.1:4);
f = 1 + 3 * w1 + w2 + 2.5 * (w1.^2 + w2.^2);
contour(w1, w2, f);
title("Trajectory of LMS");
hold on;
% Initialization
p = [1 -2; 2 1];
t = [-1 1];
alpha = 0.025;
w = [3; 1];
plot(3, 1, "r.");
% Training loop
for step = 1:40
for i = 1:2
a = dot(w', p(:, i));
error = t(i) - a;
w = w + 2 * alpha * error * p(:, i);
end
plot(w(1), w(2), "r.");
end
disp(w(1));
disp(w(2));
hold off;