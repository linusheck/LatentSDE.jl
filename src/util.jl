export Timeseries, filter_dims, map_dims, map_ts, select_ts, select_tspan, repeat_ts

"""A collection of multivariate timeseries.
This type exists to consolidate the tons of different representations that are
used for timeseries:
    - vectors of named tuples
    - vectors of RODESolutions
    - EnsembleSolutions
and work with univariate and multivariate timeseries easily, be easily
plottable and that kind of stuff.
"""
struct Timeseries{T}
    t::Vector{T}
    u::Vector{Vector{Vector{T}}}
end

function Timeseries(vec::Vector)
    t = nothing
    u = []
    for entry in vec
        if t !== nothing && entry.t != t
            throw(ArgumentError("All provided time vectors must be the same"))
        end
        t = entry.t
        if length(entry.u) > 0 && entry.u[1] isa Number
            # Univariate timeseries
            push!(u, map(x -> [x], entry.u))
        else
            # Multivariate timeseries
            push!(u, entry.u)
        end
    end
    Timeseries{eltype(t)}(t, u)
end

function Timeseries(sol::EnsembleSolution)
    Timeseries([x for x in sol])
end

function Timeseries(ts::Timeseries)
    Timeseries(ts.t, ts.u)
end

function Timeseries(timespan::Vector, matrix::Array)
    # works for a matrix like the Latent SDE returns:
    # 1 = latent space dimension
    # 2 = batch number
    # 3 = time step
    
    by_batch = map(collect, eachslice(matrix, dims=2))
    by_batch_latent = map((x) -> map(collect, eachslice(x, dims=2)), by_batch)
    
    Timeseries(timespan, by_batch_latent)
end

function Timeseries(single_element)
    Timeseries([single_element])
end

function repeat_ts(counts, ts::Timeseries)
    Timeseries(ts.t, repeat(ts.u, counts))
end

function select_ts(range, ts::Timeseries)
    Timeseries(ts.t, ts.u[range])
end

"""
Executes f on each datapoint by dimensions, so an input to f would be
    [<value in dimension 1>, <value in dimension 2>, ..., <value in dimension n>]
and f produces
    [<value in dimension 1>, <value in dimension 2>, ..., <value in dimension m>]
"""
function map_dims(f, ts::Timeseries)
    Timeseries(ts.t, map(x -> map(y -> f(y), x), ts.u))
end

"""
Removes dimensions from the timeseries by the range `dims`.
"""
function filter_dims(dims, ts::Timeseries)
    map_dims(x -> x[dims], ts)
end

"""
Executes f on each datapoint by timeseries, so an input to f would be
    [<value at dimension i in ts 1>, <value at dimension i at ts 2>, ..., <value at dimension i at ts n>]
and f produces
    [<value at dimension i in ts 1>, <value at dimension i at ts 2>, ..., <value at dimension i at ts m>]
(it gives a single timeseries per dimension)

Example: Compute mean and variance of a timeseries, then f is
f(vec) = [mean(vec), var(vec)]
"""
function map_ts(f, ts::Timeseries)
    n_ts = length(ts.u)
    n_time = length(ts.t)
    n_dims = length(ts.u[1][1])
    
    # # Probe many timeseries the function gives back
    n_ts_new = length(f(zeros(eltype(ts.t), n_ts)))
    
    result = Vector{Vector{Vector{eltype(ts.t)}}}()
    
    for i_ts in 1:n_ts_new
        push!(result, [])
        for i_time in 1:n_time
            push!(result[i_ts], [])
            for i_dims in 1:n_dims
                push!(result[i_ts][i_time], zero(eltype(ts.t)))
            end
        end
    end

    for time in 1:n_time
        for dim in 1:n_dims
            datapoint = f([ts.u[i_ts][time][dim] for i_ts in 1:n_ts])
            for ts_new in eachindex(datapoint)
                result[ts_new][time][dim] = datapoint[ts_new]
            end
        end
    end
    Timeseries(ts.t, result)
end

function select_tspan(tspan::Tuple, timeseries::Timeseries)
    t_begin = searchsortedfirst(timeseries.t, tspan[1])
    t_end = searchsortedlast(timeseries.t, tspan[2])
    Timeseries(timeseries.t[t_begin:t_end], [x[t_begin:t_end] for x in timeseries.u])
end

@recipe function f(ts::Timeseries)
    dims = length(ts.u[1][1])
    layout := @layout collect(repeat([1], dims))
    legend := false
    for dimension in 1:dims
        @series begin
            subplot := dimension
            this_dim = collect(map(case -> collect(map(el -> el[dimension], case)), ts.u))
            ts.t, this_dim
        end
    end
end

Base.:+(x::Timeseries, y::Timeseries) = Timeseries(x.t, x.u + y.u)
Base.:-(x::Timeseries, y::Timeseries) = Timeseries(x.t, x.u - y.u)
Base.:-(x::Timeseries) = Timeseries(x.t, -x.u)
