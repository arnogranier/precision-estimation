using Random;
using Distributions;
using Plots;
using LinearAlgebra;
using LaTeXStrings;

Random.seed!(1061998);
T = 1e4;
N = 100;
Ms = [1, 10, 25];
η = 0.005;

function simulate(T, N, M, η, overlap, p)
    
    # Parameters
    σ2s = [.5 .* rand(N) .+ 0.5 for _ in 1:M];
    σs = broadcast.(sqrt,σ2s);
    μs = [2*rand(N).-1 for _ in 1:M];
    if overlap
        rs = broadcast.(Int, ([rand(N) .< p for _ in 1:M]));
    else
        rs = broadcast.(Int, ([[(j==i) ? 1 : 0 for j in 1:N] for i in 1:M]));
    end
    if overlap
        A = 2 * (1. / p) .* ones(N, N)/N;
    else
        A = 2 * ones(N, N);
    end

    # Learning
    store = Vector{Float64}(undef, floor(Int, T));
    for i in 1:floor(Int, T)
        for j in 1:M
            r = rs[j]
            σ = σs[j]
            μ = μs[j]
            x = rand(MvNormal(μ, σ))
            λ = A * r
            δ = 0.5*(1 ./ λ .- (x - μ) .^ 2)
            Adot = A .* (δ * r')
            A += η .* Adot
        end
        #Evaluating
        store[i] = sum([norm(1 ./(A*rs[i]) .- σ2s[i]) for i in 1:M])/(sqrt(N)*M)
    end
    
    return store
end;

# Simulation
stores = [simulate(T, N, m, η, true, .5) for m in Ms];
append!(stores, [simulate(T, N, 100, η, false, 0)]);

# Plotting
plot(stores[1:3], linewidth=3, palette = :Dark2_4, size = (500, 400),
     label=[L"N_c=1"*" (random 50%)" L"N_c=10"*" (random 50%)" L"N_c=25"*" (random 50%)"],
     legend_font_pointsize=8, margin=3Plots.mm, thickness_scaling=4/3)
plot!(stores[4], linewidth=3, ls=:dash, c=palette(:Dark2_5)[4],
      label=L"N_c=100"*" (onehot)", legend_font_pointsize=8)
ylabel!("average second-order error")
xlabel!("time")

gui()


