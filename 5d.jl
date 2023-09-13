using Distributions;
using Random;
using Plots;

Random.seed!(123456);

σ1 = [1/4, 1];
σ2 = [1, 1/4];

ηa = 0.00075;
ηw = 0.00075;
τ = 10;

# Building dataset
μ = zeros(2);
d1 = MvNormal(μ, σ1);
d2 = MvNormal(μ, σ2);
ds = hcat([rand(d1, 5000); repeat([1,0],1,5000)],
          [rand(d2, 5000); repeat([0,1],1,5000)]);
ds = ds[:, shuffle(1:end)];

# Parameters
W = rand(2,2);
Wsave = W;
A = ones(2,2);

# Learning
k = 0
cs = [];
for e in eachcol(ds)
    d = e[1:2]
    t = e[3:4]
    λ = A * t
    e = d - W * t
    ε = λ .* e
    δ = 0.5 * (1 ./ λ - e .^ 2)
    global W += ηa * ε * t'
    global A += ηw * A .* (δ * t')
    
    # Testing
    if (k%100==0)
        c = 0
        for e in eachcol(ds)
            d = e[1:2]
            l = argmax(e[3:4])
            t = [.5, .5]
            for i in 1:10
                λ = A * t
                e = d - W * t
                ε = λ .* e
                δ = 0.5 * (1 ./ λ - e .^ 2)
                a = W' * ε + A' * δ
                t += 1/τ * (-t+a)
            end
            if argmax(t)==l
                c += 1
            end
        end
        append!(cs, c/10000)
    end

    global k+=1
end

# Learning (classical predictive coding)
W = Wsave;
k = 0;
css = [];
for e in eachcol(ds)
    d = e[1:2]
    t = e[3:4]
    e = d - W * t
    global W += ηa * e * t'

    # Testing (classical predictive coding)
    if (k%100==0)
        c_ = 0
        for e in eachcol(ds)
            d = e[1:2]
            l = argmax(e[3:4])
            t = [.5, .5]
            for i in 1:10
                e = d - W * t
                a = W' * e
                t += 1/τ * (-t+a)
            end
            if argmax(t)==l
                c_ += 1
            end
        end
        append!(css, c_/10000)
    end

    global k+=1
end 

# Maximum likelihood oracle
mle = [];
for e in eachcol(ds)
    d = e[1:2]
    t = e[3:4]
    if (k%100==0)
        c_ = 0
        for e in eachcol(ds)
            d = e[1:2]
            l = argmax(e[3:4])
            if (argmax([loglikelihood(d1, d), loglikelihood(d2, d)])) == l
                c_ += 1
            end
        end
        append!(mle, c_/10000)
    end
    global k+=1
end 

# Plotting
plot([(1 .- cs) (1 .- css) (1 .- mle)], linewidth=3, palette=:Dark2_4,
     size=(400, 500), label=["with δ-prop" "without δ-prop" "MLE oracle"],
     legend_font_pointsize=8, margin=3Plots.mm,legend=:right,
     thickness_scaling=4/3)
ylabel!("error rate")
xlabel!("epoch")

gui()