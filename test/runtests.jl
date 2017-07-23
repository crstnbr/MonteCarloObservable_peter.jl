using MonteCarloObservable
using Base.Test

# write your own tests here
@test float_observable = monte_carlo_observable{Float64}(1.0)
