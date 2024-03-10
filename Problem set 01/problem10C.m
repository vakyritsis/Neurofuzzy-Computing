% Initialization
p = [0 0 1 -1 2.1 0 1.6; 0 1 0 -1 0 -2.5 -1.6];
t = [-1 -1 -1 -1 1 1 1];
alpha = 0.04;
w = [0.5; 0.5];
b = 0.5;
% Training loop
for step = 1:36
    for i = 1:7
        a = dot(w', p(:, i)) + b;
        error = t(i) - a;
        w = w + 2 * alpha * error * p(:, i);
        b = b + 2 * alpha * error;
    end
end
disp(w(1));
disp(w(2));
disp(b);