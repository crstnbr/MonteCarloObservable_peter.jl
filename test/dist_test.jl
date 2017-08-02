using Distributions
try
    Pkg.clone("https://www.github.com/pebroecker/MonteCarloObservable")
catch
    Pkg.update("MonteCarloObservable")
end

using MonteCarloObservable

mco = monte_carlo_observable{Float64}("Scalar")
mco.keep_timeseries = true

rayleigh(x, sigma=1.0) = x / sigma^2 .* exp(-x.^2 / (2 * sigma^2))
sweep_exp = 20
sweeps = 2^sweep_exp
x = rand() * 5
xs = zeros(sweeps)

# thermalization
for j in 1:sweeps / 8
    r = rand() * 5
    if rand() < rayleigh(r) / rayleigh(x)   x = r   end
end

for j in 1:sweeps
    r = rand() * 5
    if rand() < rayleigh(r) / rayleigh(x)   x = r   end

    push!(mco, x)
    xs[j] = x
end

println("Mean of x: $(mean(mco))")
println("Variance of x: $(var(mco))")
println("Binning error of x $(binning_error(mco))")
println("Jackknife error of x $(jackknife_error(mco))")
