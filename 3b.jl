using Random;
using Distributions;
using Plots;
using LinearAlgebra;
using LaTeXStrings;

Random.seed!(1061998);

T = 1e4;
N = 100;
Ms = [1, 10, 25];
η = 0.00006;

function simulatemean(T, N, M, η, overlap, p)

    # Parameters
    σ2s = [1.5 .* rand(N) .+ 0.5 for _ in 1:M] ./ 2;
    σs = broadcast.(sqrt,σ2s);
    μs = [2*rand(N).-1 for _ in 1:M];
    if overlap
        rs = broadcast.(Int, ([rand(N) .< p for _ in 1:M]));
    else
        rs = broadcast.(Int, ([[(j==i) ? 1 : 0 for j in 1:N] for i in 1:M]));
    end

    # Learning
    store = Vector{Float64}(undef, floor(Int, T));
    W = zeros(N, N);
    for i in 1:floor(Int, T)
        for j in 1:M
            r = rs[j]
            σ = σs[j]
            μ = μs[j]
            x = rand(MvNormal(μ, σ))
            λ = 1
            ε = λ .* (x-W*r)
            Wdot = (ε * r')
            W += η .* Wdot
        end
        # Evaluating
        store[i] = sum([norm(W*rs[i] .- μs[i]) for i in 1:M])/(sqrt(N)*M)
    end
    
    return store
end;

# Simulation
stores = [simulatemean(T, N, m, η, true, .5) for m in Ms];
append!(stores, [simulatemean(T, N, 100, 25*η, false, 0)]);

# Plotting
plot(stores[1:3], linewidth=3, palette=:Dark2_4, size=(500, 400),
     label=[L"N_c=1"*" (random 50%)" L"N_c=10"*" (random 50%)" L"N_c=25"*" (random 50%)"],
     legend_font_pointsize=8, margin=3Plots.mm, thickness_scaling=4/3)
plot!(stores[4], linewidth=3, linestyle=:dash, c=palette(:Dark2_5)[4],
      label=L"N_c=100"*" (onehot)", legend_font_pointsize=8)
ylabel!("average prediction error")
xlabel!("time")

gui()
