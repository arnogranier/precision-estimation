using Distributions;
using Random;
using Plots;
using LinearAlgebra;
Random.seed!(123456);

d1 = 100;
d0 = 100;
Nc = 100;
τ = 200;

# Parameters
σ21 = [[rand((0.1, 2.)) for _ in 1:d1] for i in 1:Nc];
μ1 = [rand(d1)/50 for i in 1:Nc];
σ21bar = mean(σ21);
A = 3rand(d1, d1);
λ0bar = mean([A*μ for μ in μ1]);
W = I(d1);
errs = [[], [], [], []];

# Main Loop
for i = 1:Nc
    err1, err2, err3, err4 = 0, 0, 0, 0;

    # Sample
    λ1 = 1 ./ σ21[i]
    D1 = MvNormal(μ1[i], broadcast.(sqrt, σ21)[i])
    x = rand(D1)
    λ0 = A*max.(x, 0)
    D2 = MvNormal(x, sqrt.(1 ./ λ0))
    data = rand(D2)

    # Bayes-optimal estimate
    xhat = (λ1 .* μ1[i] + λ0 .* data) ./ (λ1+λ0)
    err1 += norm(x-xhat) ./ sqrt(d1)
    append!(errs[1], err1/Nc)

    # precision estimates
    u = zeros(d1) .+ 1
    for t in 1:2000
        λhat = A*max.(u,0)
        u += (1/τ) * (-u + μ1[i] +  σ21[i] .* (λhat .* (W' * (data - W*u))))
    end
    err2 += norm(x-u) ./ sqrt(d1)
    append!(errs[2], err2/Nc)

    # mean precision
    u = zeros(d1) .+ 1
    for t in 1:2000
        u += (1/τ) * (-u + μ1[i] + σ21bar .* λ0bar .* W'*(data - W*u))
    end
    err3 += norm(x-u) ./ sqrt(d1)
    append!(errs[3], err3/Nc)

    # no weighting
    u = zeros(d1) .+ 1
    for t in 1:2000
        u += (1/τ) * (-u + μ1[i] + W' * (data - W*u))
    end
    err4 += norm(x-u) ./ sqrt(d1)
    append!(errs[4], err4/Nc)
end

# Plotting
c = palette(:Dark2_4);
Plots.bar([1,], [100 .* mean(errs[1]),], fillcolor=c[1],
          yerr=[100 .* std(errs[1]),], legend_font_pointsize=8,
          margin=3Plots.mm, thickness_scaling=4/3, size=(370, 400),
          label="Bayes-optimal", xticks=false)
Plots.bar!([2,], [100 .* mean(errs[2]),], fillcolor=c[2],
           label="precision estimates", yerr=[100 .* std(errs[2]),],
           xticks=false)
Plots.bar!([3,], [100 .* mean(errs[3]),], fillcolor=c[3],
           label="mean precision", yerr=[100 .* std(errs[3]),], xticks=false)
Plots.bar!([4,], [100 .* mean(errs[4]),], fillcolor=c[4], label="no weighting",
           yerr=[100 .* std(errs[4]),], xticks=false)
ylabel!("average error")

gui()
