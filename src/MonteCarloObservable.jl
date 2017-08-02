module MonteCarloObservable

    using HDF5

    include("ObservableType.jl")
    include("ObservableStatistics.jl")
    include("ObservableIO.jl")

    export MonteCarloObservable

    function Base.start(mco::monte_carlo_observable) state = 1 end
    function Base.done(mco::monte_carlo_observable, state::Int) return state == mco.curr_bin end
    function Base.next(mco::monte_carlo_observable, state::Int)
        return mco.bins[mco.colons..., state], state + 1
    end
end
