function extension_acf_v4
rng(1)

N_total = 50000;
burnin  = 5000;
maxLag  = 500;

sigmas  = [0.01, 0.05, 0.2, 2.0, 10.0];
x0      = 0.5;

results = zeros(numel(sigmas), 5);
chains  = cell(numel(sigmas),1);

figure; hold on
for j = 1:numel(sigmas)
    sigma = sigmas(j);

    [chain, accRate] = run_mh_1d(N_total, x0, sigma);

    chain_post = chain(burnin+1:end);
    n = numel(chain_post);
    chains{j} = chain_post;

    acfVals = acf_fft(chain_post, maxLag);

    tauInt = 1;
    for k = 1:maxLag
        if acfVals(k+1) < 0
            break
        end
        tauInt = tauInt + 2*acfVals(k+1);
    end

    ESS = n / tauInt;

    dx = diff(chain_post);
    ESJD = mean(dx.^2);

    results(j,:) = [sigma, accRate, tauInt, ESS, ESJD];

    plot(0:maxLag, acfVals, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\sigma=%.3g', sigma));
end
hold off
xlabel('Lag')
ylabel('Autocorrelation')
title('ACF for different proposal step sizes')
legend('Location','northeast')
save_fig('acf_compare.png')

T = array2table(results, ...
    'VariableNames', {'sigma','acceptRate','tau_int','ESS','ESJD'});
disp('MH efficiency summary (post burn-in)')
disp(T)

trace_sigmas = [0.01, 0.2, 2.0, 10.0];
idx = arrayfun(@(s) find(abs(sigmas-s)<1e-12,1), trace_sigmas);

figure
for k = 1:numel(idx)
    subplot(numel(idx),1,k)
    plot(chains{idx(k)}, 'LineWidth', 0.7); ylim([0 1])
    title(sprintf('Trace plot (post burn-in), \\sigma=%.2f', trace_sigmas(k)))
    xlabel('Iteration'); ylabel('X')
end
save_fig('trace_compare.png')

best_sigma = 0.2;
best_idx = find(abs(sigmas - best_sigma) < 1e-12, 1);
x_samples = chains{best_idx};

xx = linspace(0,1,500);
pp = arrayfun(@target_unnorm, xx);
pp = pp / trapz(xx, pp);

figure
histogram(x_samples, 40, 'Normalization','pdf')
hold on
plot(xx, pp, 'LineWidth', 2)
hold off
xlabel('x'); ylabel('Density')
title(sprintf('Histogram vs target density (\\sigma=%.2f)', best_sigma))
legend('MH samples','Target density', 'Location','northeast')
save_fig('hist_target.png')

figure
yyaxis left
plot(results(:,1), results(:,5), '-o', 'LineWidth', 1.5)
ylabel('ESJD')
yyaxis right
plot(results(:,1), results(:,2), '-s', 'LineWidth', 1.5)
ylabel('Acceptance rate')
xlabel('\sigma')
title('Goldilocks principle: too small vs too large step size')
save_fig('goldilocks.png')

end

function [samples, accRate] = run_mh_1d(N, x0, sigma)
samples = zeros(N,1);
x = x0;
acc = 0;
px = target_unnorm(x);

for i = 1:N
    x_prop = x + sigma*randn();
    p_prop = target_unnorm(x_prop);

    if px == 0
        alpha = 1;
    else
        alpha = min(1, p_prop / px);
    end

    if rand < alpha
        x = x_prop;
        px = p_prop;
        acc = acc + 1;
    end

    samples(i) = x;
end

accRate = acc / N;
end

function acfVals = acf_fft(x, maxLag)
x = x(:);
x = x - mean(x);
n = numel(x);

m = 2^nextpow2(2*n);
X = fft(x, m);
S = X .* conj(X);
c = ifft(S);
c = real(c(1:maxLag+1));

c = c ./ (n:-1:n-maxLag)';
acfVals = c / c(1);
acfVals(1) = 1;
end

function p = target_unnorm(x)
if x < 0 || x > 1
    p = 0;
    return
end
p = exp(-5*(x-0.5)^2) * (cos(3*pi*x) + 1);
end

function save_fig(filename)
try
    exportgraphics(gcf, filename, 'Resolution', 300);
catch
    saveas(gcf, filename);
end
end