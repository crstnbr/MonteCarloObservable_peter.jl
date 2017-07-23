module MonteCarloObservable

    using HDF5

    type monte_carlo_observable{T}
        name::String
        n_measurements::Int
        keep_timeseries::Int
        entry_dims::Array{Int, 1}
        entry_size::Int
        timeseries::Array{T}
        measurement_buffer::Array{T}
        bins::Array{T}
        curr_bin::Int
        autocorrelation_buffer::Array{T}
        monte_carlo_observable(name::String, entry_dims::Array{Int, 1}=[1]) = new(name, 1, false, entry_dims, prod(entry_dims), typemin(T) * ones(entry_dims..., 1), typemin(T) * ones(entry_dims..., 1), typemin(T) * ones(entry_dims..., 2^8), 1, typemin(T) * ones(T, entry_dims..., 2^10))
    end

    """
    Adding a measurement to the observable triggers the following cascade:

    1. the measurement is added to the buffers
    2. if the buffer is full
        - all measurements are pushed to the timeseries array if desired
        - buffer is averaged and added to bins
            + if all bins are used -> rebin, adjust buffer size
    """
    function Base.push!{T}(mco::monte_carlo_observable{T}, measurement::T, verbose = false)
        push!(mco, T[measurement], verbose)
    end

    function Base.push!{T}(mco::monte_carlo_observable{T}, measurement::Array{T}, verbose = false)
        colons = [Colon() for _ in mco.entry_dims]
        buffer_size = size(mco.measurement_buffer)[end]
        ac_buffer_size = size(mco.autocorrelation_buffer)[end]
        bin_size = size(mco.bins)[end]

        measurement_entry = mod1(mco.n_measurements, buffer_size)
        autocorrelation_entry = mod1(mco.n_measurements, ac_buffer_size)
        if verbose println("Saving measurement to position $(measurement_entry)") end

        mco.measurement_buffer[colons..., measurement_entry][:] = measurement
        mco.autocorrelation_buffer[colons..., autocorrelation_entry][:] = measurement

        if mod(mco.n_measurements, buffer_size) == 0
            if verbose println("Buffer is full") end
            if mco.keep_timeseries == true
                if verbose println("Appending to timeseries") end
                timeseries_copy = copy(mco.timeseries)
                mco.timeseries = Array{T}(mco.entry_dims..., mco.n_measurements)
                mco.timeseries[colons..., 1:size(timeseries_copy)[end]] = timeseries_copy
                mco.timeseries[colons..., size(timeseries_copy)[end] + 1:end] = mco.measurement_buffer
            end

            bin_entry = mod1(mco.n_measurements, bin_size)
            if verbose println("Calculating current bin $(mco.curr_bin)") end
            mco.bins[colons..., mco.curr_bin] = mean(mco.measurement_buffer, length(size(mco.measurement_buffer)))

            if mco.curr_bin == bin_size
                if verbose println("All bins are full") end
                for (i, j) in enumerate(1:2:bin_size)
                    mco.bins[colons..., i] = 0.5 * (mco.bins[colons..., j] + mco.bins[colons..., j + 1])
                end

                mco.curr_bin = Int(0.5 * bin_size) + 1
                if verbose println("Starting refill at $(mco.curr_bin)" ) end
                mco.bins[colons..., mco.curr_bin:end] = typemin(T)

                mco.measurement_buffer = typemin(T) * ones(mco.entry_dims..., 2 * buffer_size)
                if verbose println("New buffer size $(size(mco.measurement_buffer)[end])" ) end
            else
                mco.curr_bin += 1
            end
        end
        mco.n_measurements += 1
    end

    function HDF5.write{T}(h5file::HDF5File, mco::monte_carlo_observable{T})
        write_parameters(h5file, mco)
        write_datasets(h5file, mco)
    end

    function write_parameters{T}(h5file::HDF5File, mco::monte_carlo_observable{T})
        grp_prefix = "simulation/results/$(mco.name)"

        if exists(h5file, grp_prefix)
            o_delete(h5file, "$(grp_prefix)/keep_timeseries")
            o_delete(h5file, "$(grp_prefix)/n_measurements")
            o_delete(h5file, "$(grp_prefix)/curr_bin")
        end

        h5file["$(grp_prefix)/keep_timeseries"] = mco.keep_timeseries
        h5file["$(grp_prefix)/n_measurements"] = mco.n_measurements
        h5file["$(grp_prefix)/curr_bin"] = mco.curr_bin
    end

    # function write_scalar_datasets{T}(h5file::HDF5File, mco::monte_carlo_observable{T})
    #     grp_prefix = "simulation/results/$(mco.name)"
    #
    #     timeseries_size = ((1024, ), (-1, ))
    #     buffer_size = ((1024, ), (-1, ))
    #     chunk_size = (256,)
    #
    #     if exists(h5file, "$(grp_prefix)/measurement_buffer")
    #         println("Resizing ", (size(mco.measurement_buffer, 1), ))
    #
    #         set_dims!(h5file["$(grp_prefix)/measurement_buffer"], (size(mco.measurement_buffer, 1), ))
    #         h5file["$(grp_prefix)/measurement_buffer"][:] = mco.measurement_buffer[:]
    #         h5file["$(grp_prefix)/bins"][:] = mco.bins[:]
    #
    #         if size(mco.timeseries, 1) > 1
    #             set_dims!(h5file["$(grp_prefix)/timeseries"], (size(mco.timeseries, 1), 1))
    #             h5file["$(grp_prefix)/timeseries"][:] = mco.timeseries[:]
    #         end
    #     else
    #         m_set = d_create(h5file, "$(grp_prefix)/measurement_buffer", T, ((size(mco.measurement_buffer, 1),), (-1,)), "chunk", chunk_size)
    #         m_set[:] = mco.measurement_buffer[:]
    #         h5file["$(grp_prefix)/bins"] = mco.bins
    #
    #         if size(mco.timeseries, 1) > 1
    #             t_set = d_create(h5file, "$(grp_prefix)/timeseries", T, ((size(mco.timeseries, 1),), (-1,)), "chunk", chunk_size)
    #             t_set[:] = mco.timeseries[:]
    #         end
    #     end
    # end

    function write_datasets{T}(h5file::HDF5File, mco::monte_carlo_observable{T})
        colons = [Colon() for _ in mco.entry_dims]
        grp_prefix = "simulation/results/$(mco.name)"
        timeseries_size = (size(mco.timeseries), (mco.entry_dims..., -1))
        buffer_size = (size(mco.measurement_buffer), (mco.entry_dims..., -1))
        chunk_size = (mco.entry_dims..., 256)

        if exists(h5file, "$(grp_prefix)/measurement_buffer")
            set_dims!(h5file["$(grp_prefix)/measurement_buffer"], size(mco.measurement_buffer))
            h5file["$(grp_prefix)/measurement_buffer"][colons..., 1:size(mco.measurement_buffer)[end]] = mco.measurement_buffer
            h5file["$(grp_prefix)/bins"][colons..., :] = mco.bins
            h5file["$(grp_prefix)/autocorrelation_buffer"][colons..., :] = mco.autocorrelation_buffer
            set_dims!(h5file["$(grp_prefix)/timeseries"], size(mco.timeseries))
            h5file["$(grp_prefix)/timeseries"][colons..., 1:size(mco.timeseries)[end]]  = mco.timeseries[:]
        else
            m_set = d_create(h5file, "$(grp_prefix)/measurement_buffer", T, buffer_size, "chunk", chunk_size)
            m_set[colons..., 1:size(mco.measurement_buffer)[end]] = mco.measurement_buffer
            h5file["$(grp_prefix)/bins"] = mco.bins
            h5file["$(grp_prefix)/autocorrelation_buffer"] = mco.autocorrelation_buffer
            t_set = d_create(h5file, "$(grp_prefix)/timeseries", T, timeseries_size, "chunk", chunk_size)
            t_set[colons..., 1:size(mco.timeseries)[end]]  = mco.timeseries
        end
    end

    function HDF5.read!{T}(h5file::HDF5File, mco::monte_carlo_observable{T})
        grp_prefix = "simulation/results/$(mco.name)"

        if exists(h5file, grp_prefix)
            println("Reading")
            mco.n_measurements = read(h5file, "$(grp_prefix)/n_measurements")
            mco.keep_timeseries = read(h5file, "$(grp_prefix)/keep_timeseries")
            mco.timeseries = read(h5file, "$(grp_prefix)/timeseries")
            mco.measurement_buffer = read(h5file, "$(grp_prefix)/measurement_buffer")
            mco.bins = read(h5file, "$(grp_prefix)/bins")
            mco.curr_bin = read(h5file, "$(grp_prefix)/curr_bin")
            println("curr bin after loading is $(mco.curr_bin)")
            mco.autocorrelation_buffer = read(h5file, "$(grp_prefix)/autocorrelation_buffer")
        end
    end
end
