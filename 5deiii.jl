using Distributions;
using Random;
using Plots;

Random.seed!(123456);

σ = [1/5, 1/5];
μ1 = [1,0];
μ2 = [-1,0];
μ3 = [0,1];
μ4 = [0,-1];

N = 1000;
ηa = 0.001;
ηw = 0.001;
τ = 100;
T = 300;

# Building dataset
d1 = MvNormal(μ1, σ);
d2 = MvNormal(μ2, σ);
d3 = MvNormal(μ3, σ);
d4 = MvNormal(μ4, σ);
ds = hcat([[rand(d1, N÷2) rand(d2, N÷2)]; repeat([1,0],1,N)],
          [[rand(d3, N÷2) rand(d4, N÷2)]; repeat([0,1],1,N)]);
ds = ds[:, shuffle(1:end)];

# Parameters
W = rand(2,2);
A = ones(2,2);
relu(x) = x .* Float64.(x .> 0);
drelu(x) = Float64.(x .> 0)

# Learning
for epoch in 1:100
    for e in eachcol(ds)
        d = e[1:2]
        t = e[3:4]
        λ = A * t
        e = d - W * t
        ε = λ .* e
        δ = 0.5 * (1 ./ λ - e .^ 2)
        global W += ηa * ε * t'
        global A += ηw * A .* (δ * t')
    end
end

# Inference
cs = [];
for e in eachcol(ds)
    d = e[1:2]
    t = [.5, .5]
    for i in 1:T
        λ = A * relu(t)
        e = d - W * relu(t)
        ε = λ .* e
        δ = 0.5 * (1 ./ λ - e .^ 2)
        a = drelu(t) .* (W' * ε + A' * δ)
        t += 1/τ * (-t+a)
    end
    append!(cs, argmax(t))
end

# Plotting
scatter(ds[1,:], ds[2,:], color=cs,palette=["blue", "red"], axis=nothing,
        legend=nothing, border=:none, aspect_ratio=:equal, layout=(2,1))


# Learning (classical predictive coding)
W = rand(2,2);
for epoch in 1:100
    for e in eachcol(ds)
        d = e[1:2]
        t = e[3:4]
        e = d - W * t
        global W += ηa * e * t'
    end
end

# Inference (classical predictive coding)
cs = [];
for e in eachcol(ds)
    d = e[1:2]
    t = [.5, .5]
    for i in 1:T
        e = d - W * relu(t)
        a = drelu(t) .* (W' * e )
        t += 1/τ * (-t+a)
    end
    append!(cs, argmax(t))
end

# Plotting
scatter!(ds[1,:], ds[2,:], color=cs,palette=["blue", "red"], axis=nothing,
        legend=nothing, border=:none, aspect_ratio=:equal, subplot=2)

gui()