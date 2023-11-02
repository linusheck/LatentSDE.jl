export LatentNeuralSDE, StandardLatentSDE, sample_prior, sample_prior_dataspace

mutable struct LatentNeuralSDE{N1,N2,N3,N4,N5,N6,N7,N8,S,T,D,K} <: LuxCore.AbstractExplicitContainerLayer{(:initial_prior, :initial_posterior, :drift_prior, :drift_posterior, :diffusion, :encoder_recurrent, :encoder_net, :projector,)}
    initial_prior::N1
    initial_posterior::N2
    drift_prior::N3
    drift_posterior::N4
    diffusion::N5
    encoder_recurrent::N6
    encoder_net::N7
    projector::N8
    solver::S
    tspan::T
    datasize::D
    kwargs::K
end

function LatentNeuralSDE(initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder_recurrent, encoder_net, projector, solver, tspan, datasize; kwargs...)
    models = [initial_prior, initial_posterior, drift_prior, drift_posterior, diffusion, encoder_recurrent, encoder_net, projector]
    LatentNeuralSDE{
        [typeof(x) for x in models]...,
        typeof(solver),typeof(tspan),typeof(datasize),typeof(kwargs)
    }(
        models...,
        solver, tspan, datasize, kwargs
    )
end

"""
    StandardLatentSDE(solver, tspan, datasize)
    
Constructs a "standard" latent sde - so you don't need to construct all of the neural nets.

## Arguments

- `solver`: SDE solver.
- `tspan`: SDE timespan.
- `datasize`: SDE timesteps.

## Keyword Arguments

- `data_dims`: Dimension of the data space.
- `latent_dims`: Dimension of the latent space.
- `prior_size`: Size of the prior net hidden layers.
- `posterior_size`: Size of the posterior net hidden layers.
- `diffusion_size`: Size of the diffusion hidden layers. The diffusion is not
    super configurable. Please just swap it out if you need a different one. Prior
- `depth`: Depth of the prior and posterior nets.
- `hidden_activation`: Activation of the hidden layers of the neural nets.
- `rnn_size`: Size of the RNN's output. There's a neural net after the RNN that goes to `context_size`.
- `context_size`: Size of the context vector.
"""
function StandardLatentSDE(solver, tspan, datasize;
        data_dims=1,
        latent_dims=2,
        prior_size=64,
        posterior_size=64,
        diffusion_size=16,
        depth=1,
        rnn_size=16,
        context_size=16,
        hidden_activation=tanh,
        kwargs...
    )
    
    solver_kwargs = Dict(kwargs)
    networks = []

    # this function serves the following purpose:
    # - create default network if there isn't one given in the kwargs
    # - remove that kwarg from the kwargs given to the solver
    # needs to be called in the same order as args to the LatentNeuralSDE
    function create_network(name, network)
        if haskey(solver_kwargs, name)
            # network is provided by user
            user_network = pop!(solver_kwargs, name)
            push!(networks, user_network)
        else
            push!(networks, network)
        end
    end

    in_dims = latent_dims

    # The initial_posterior net is the posterior for the initial state. It
    # takes the context and outputs a mean and standard devation for the
    # position zero of the posterior. The initial_prior is a fixed gaussian
    # distribution.
    create_network(:initial_prior, Lux.Dense(1 => latent_dims + latent_dims, use_bias=false, init_weight=Lux.zeros32))
    create_network(:initial_posterior, Lux.Dense(context_size => latent_dims + latent_dims, init_weight=Lux.zeros32, init_bias=Lux.zeros32))
    
    # Drift of prior. This is just an SDE drift in the latent space
    create_network(:drift_prior, Lux.Chain(
        Lux.Dense(in_dims => prior_size, hidden_activation),
        repeat([Lux.Dense(prior_size => prior_size, hidden_activation)], depth)...,
        Lux.Dense(prior_size => latent_dims, hidden_activation)
    ))
    # Drift of posterior. This is the term of an SDE when fed with the context.
    create_network(:drift_posterior, Lux.Chain(
        Lux.Dense(in_dims + context_size => posterior_size, hidden_activation),
        repeat([Lux.Dense(posterior_size => posterior_size, hidden_activation)], depth)...,
        Lux.Dense(posterior_size => latent_dims, hidden_activation)
    ))
    # Prior and posterior share the same diffusion (they are not actually evaluated
    # seperately while training, only their KL divergence). This is a diagonal
    # diffusion, i.e. every term in the latent space has its own independent
    # Wiener process.
    create_network(:diffusion, Diagonal([
            Lux.Chain(
                Lux.Dense(1 => diffusion_size, hidden_activation),
                Lux.Dense(diffusion_size => diffusion_size, hidden_activation),
                Lux.Dense(diffusion_size => 1, Lux.sigmoid),
                Lux.WrappedFunction(Base.Fix1(broadcast, (x) -> 10f0 * x + 1f-5))
            ) for i in 1:latent_dims]...)
    )

    # The encoder is a recurrent neural network.
    create_network(:encoder_recurrent, Lux.Recurrence(Lux.GRUCell(data_dims => rnn_size); return_sequence=true))
    
    # The encoder_net is appended to the results of the encoder. Couldn't make
    # this work directly in Lux.
    create_network(:encoder_net, Lux.Dense(rnn_size => context_size))

    # The projector will transform the latent space back into data space.
    create_network(:projector, Lux.Dense(latent_dims => data_dims))

    return LatentNeuralSDE(
        networks...,
        solver,
        tspan,
        datasize;
        solver_kwargs...
    )
end

function get_distributions(model, model_p::ComponentArray{Float32}, st, context::AbstractArray{Float32})
    normsandvars, _ = model(context, model_p, st)
    dim_1 = size(normsandvars)[1]
    # exp the variance because it should be > 0
    return normsandvars[1:dim_1÷2, :], exp.(normsandvars[dim_1÷2+1:dim_1, :])
end

function sample_prior(n::LatentNeuralSDE, ps, st; b=1, seed=nothing, noise=(seed) -> nothing, tspan=n.tspan, datasize=n.datasize)
    (eps, seed) = ChainRulesCore.ignore_derivatives() do
        if seed !== nothing
            Random.seed!(seed)
        end
        latent_dimensions = n.initial_prior.out_dims ÷ 2
        (rand(Normal{Float32}(0.0f0, 1.0f0), (latent_dimensions, b)) |> Lux.gpu, rand(UInt32))
    end

    # We vcat 0f0 to these so that the prior has the same dimensions as the posterior (for noise reasons)
    function dudt_prior(u, p, t) 
        vcat(n.drift_prior(u[1:end-1], p.drift_prior, st.drift_prior)[1], 0f0)
    end
    function dudw_diffusion(u, p, t) 
        vcat(n.diffusion(u[1:end-1], p.diffusion, st.diffusion)[1], 0f0)
    end

    initialdists_prior_norms, initialdists_prior_vars = get_distributions(n.initial_prior, ps.initial_prior, st.initial_prior, [1.0f0] |> Lux.gpu)

    z0 = initialdists_prior_norms .+ eps .* initialdists_prior_vars
    function prob_func(prob, batch, repeat)
        noise_instance = ChainRulesCore.ignore_derivatives() do
            noise(Int(floor(seed + batch)))
        end
        SDEProblem{false}(dudt_prior, dudw_diffusion, vcat(z0[:, batch], 0f0), tspan |> Lux.gpu, ps, seed=seed + batch, noise=noise_instance)
    end

    ensemble = EnsembleProblem(nothing, output_func=(sol, i) -> (sol, false), prob_func=prob_func)

    Timeseries(solve(ensemble, n.solver, trajectories=b; saveat=range(tspan[1], tspan[end], datasize), dt=(tspan[end] / datasize), n.kwargs...))
end

function sample_prior_dataspace(n::LatentNeuralSDE, ps, st; kwargs...)
    prior_latent = sample_prior(n, ps, st; kwargs...)
    map_dims(x -> n.projector(x[1:end-1] |> Lux.gpu, ps.projector |> Lux.gpu, st.projector |> Lux.gpu)[1] |> Lux.cpu, prior_latent)
end

# from https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
function stable_divide(a::AbstractArray{Float32}, b::AbstractArray{Float32}, eps=1f-5)
    ChainRulesCore.ignore_derivatives() do
        if any(map(x -> abs(x) <= eps, b))
            @warn "diffusion too small"
        end
    end
    b = map(x -> abs(x) <= eps ? eps * sign(x) : x, b)
    a ./ b
end

function augmented_drift_batch(n::LatentNeuralSDE, times::AbstractArray{Float32}, latent_dims::Int, batch_size::Int, st::NamedTuple, u_in_vec::AbstractArray{Float32}, info::ComponentVector{Float32}, t::Float32)
    u_in = reshape(u_in_vec, latent_dims + 1, batch_size)
    u = u_in[1:end-1, :]

    # Remove augmented term from input
    p = info.ps
    context = info.context

    # Get the context for the posterior at the current time
    # initial state evolve => get the posterior at future start time
    time_index = max(1, searchsortedlast(times, t))
    posterior_net_input::AbstractArray{Float32} = vcat(u, context[:, :, time_index])

    posterior = n.drift_posterior(posterior_net_input, p.drift_posterior, st.drift_posterior)[1]
    prior = n.drift_prior(u, p.drift_prior, st.drift_prior)[1]

    # The diffusion is diagonal, so a single network is invoked on each dimension
    diffusion = n.diffusion(u, p.diffusion, st.diffusion)[1]

    # The augmented term for computing the KL divergence
    u_term = (posterior .- prior) ./ diffusion
    augmented_term = 0.5f0 .* sum(abs2, u_term; dims=1)

    reshape(vcat(posterior, augmented_term), (latent_dims + 1) * batch_size)
end

function augmented_diffusion_batch(n::LatentNeuralSDE, latent_dims::Int, batch_size::Int, st::NamedTuple, u_in_vec::AbstractArray{Float32}, info::ComponentVector{Float32}, t::Float32)
    p = info.ps
    u_in = reshape(u_in_vec, latent_dims + 1, batch_size)
    u = u_in[1:end-1, :]

    diffusion = n.diffusion(u, p.diffusion, st.diffusion)[1]
    
    cat_this = ChainRulesCore.@ignore_derivatives zeros32((1, batch_size)) |> Lux.gpu

    reshape(vcat(diffusion, cat_this), (latent_dims + 1) * batch_size)
end

# Reference:
# KL(P::Distributions.Normal, Q::Distributions.Normal) = log(Q.σ / P.σ) + (1/2) * ((P.σ / Q.σ)^2 + (P.μ - Q.μ)^2 * Q.σ^(-2) -1.)
KL(p_norms, p_vars, q_norms, q_vars) = 
    log.(q_vars ./ p_vars) .+ # log(Q.σ / P.σ) +
    (0.5f0) .* ((p_vars ./ q_vars).^2 .+ # (1/2) * ((P.σ / Q.σ)^2 +
    (p_norms .- q_norms).^2 .* q_vars.^(-2) .- 1f0) # (P.μ - Q.μ)^2 * Q.σ^(-2) -1.)

loglike(means, vars, obs) = -log.(vars) .- 0.5f0 .* log.(2f0 .* pi) .- 0.5f0 .* ((obs - means) ./ vars) .^2

"""
    (n::LatentNeuralSDE)(timeseries, ps::ComponentVector, st)
    
Sample from the Latent SDE's posterior, compute the KL-divergence and the terms for the loss.

If the model's `tspan` starts earlier than the `timeseries`, the initial states are not directly scored.
Autodiff this function to train the Latent SDE.

## Arguments

- `sense`: Sensitivity Algorithm. Consult https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/
- `seed`: Seed for simulations, we use `seed`, `seed + 1`, `seed + 2`, and so on. If no seed is provided, it's generated by the global RNG.
- `noise`: Function from `seed` to the noise used.
- `likelihood_dist`: Distribution for computing log-likelihoods (as a way of computing distance).
- `likelihood_scale`: Variance of `likelihood_dist`.
"""
function (n::LatentNeuralSDE)(timeseries::Timeseries, ps::ComponentVector, st;
    sense=InterpolatingAdjoint(autojacvec=ZygoteVJP(), checkpointing=false),
    seed=nothing,
    noise=(seed, noise_size) -> nothing,
    likelihood_dist=Normal,
    likelihood_scale=0.01f0,
)
    latent_dimensions = n.initial_prior.out_dims ÷ 2
    batch_size = length(timeseries.u)
    time_steps = length(timeseries.t)
    tspan = (timeseries.t[1], timeseries.t[end])
    # regularily sample the latent sde
    dt = (timeseries.t[end] - timeseries.t[1]) / time_steps

    # We are using matrices with the following dimensions:
    # 1 = latent space dimension
    # 2 = batch number
    # 3 = time step
    (eps, seed) = ChainRulesCore.ignore_derivatives() do
        if seed !== nothing
            Random.seed!(seed)
        end
        (rand(Normal{Float32}(0.0f0, 1.0f0), (latent_dimensions, batch_size)) |> Lux.gpu, rand(UInt32))
    end

    tsmatrix = ChainRulesCore.ignore_derivatives() do
        reduce(hcat, [reshape(map(only, u), 1, 1, :) for u in timeseries.u]) |> Lux.gpu
    end

    timecat = ChainRulesCore.ignore_derivatives() do
        function(x, y)
            cat(x, y; dims=3)
        end
    end

    # Lux recurrent uses batches / time the other way around...
    # time: dimension 3 => dimension 2
    # batches: dimension 2 => dimension 3
    tsmatrix_flipped = ChainRulesCore.ignore_derivatives() do
        reverse(permutedims(tsmatrix, (1, 3, 2)), dims=2)
    end
    
    precontext_flipped = reverse(n.encoder_recurrent(tsmatrix_flipped, ps.encoder_recurrent, st.encoder_recurrent)[1])
    context_flipped = [n.encoder_net(x, ps.encoder_net, st.encoder_net)[1] for x in precontext_flipped]

    # context_flipped is now a vector of 2-dim matrices
    # latent space: dimension 1
    # batch: dimension 2
    context_precomputed = reduce(timecat, context_flipped)

    initialdists_prior_norms, initialdists_prior_vars = get_distributions(n.initial_prior, ps.initial_prior, st.initial_prior, ChainRulesCore.ignore_derivatives(() -> [1f0] |> Lux.gpu))

    initialdists_posterior_norms, initialdists_posterior_vars = get_distributions(n.initial_posterior, ps.initial_posterior, st.initial_posterior, context_precomputed[:, :, 1])
    
    z0 = initialdists_posterior_norms .+ eps .* initialdists_posterior_vars

    augmented_z0 = vcat(z0, zeros32(1, length(z0[1, :])))

    # Deriving operations with ComponentArray is not so easy, first we have to
    # grab the axes and then re-construct using a vector
    axes = ChainRulesCore.ignore_derivatives() do
        getaxes(ComponentArray((context=context_precomputed |> Lux.cpu, ps=ps |> Lux.cpu)))
    end

    vec_context = vec(context_precomputed)

    info = ComponentArray(vcat(vec_context, ps), axes)

    noise_instance = ChainRulesCore.ignore_derivatives() do
        noise(Int(floor(seed)), (latent_dimensions + 1) * batch_size)
    end
    u0 = reshape(augmented_z0, (latent_dimensions + 1) * batch_size)
    
    sde_problem = SDEProblem{false}(
        (u, p, t) -> augmented_drift_batch(n, timeseries.t, latent_dimensions, batch_size, st, u, p, t),
        (u, p, t) -> augmented_diffusion_batch(n, latent_dimensions, batch_size, st, u, p, t),
        u0,
        tspan,
        info,
        seed=seed,
        noise=noise_instance
    )
    solution = solve(sde_problem, n.solver; sensealg=sense, saveat=collect(range(tspan[1], tspan[end], time_steps)), dt=dt, n.kwargs...)

    ts_indices = ChainRulesCore.ignore_derivatives() do
        [searchsortedfirst(solution.t, t) for t in timeseries.t]
    end
    
    #return nothing, nothing, nothing, 0f0, sum(solution_cu_array)

    #sol = reduce(timecat, map(x -> reshape(x, latent_dimensions + 1, batch_size), solution.u))
    
    solution_cu_array = solution |> Lux.gpu
    sol = reshape(vec(solution_cu_array), latent_dimensions + 1, batch_size, time_steps)

    posterior_latent = sol[1:end-1, :, :]
    kl_divergence_time = sol[end:end, :, :]

    initialdists_kl = KL(initialdists_posterior_norms, initialdists_posterior_vars, repeat(initialdists_prior_norms, 1, length(timeseries.u)), repeat(initialdists_prior_vars, 1, length(timeseries.u)))
    kl_divergence = (sum(initialdists_kl, dims=1) .+ sol[end:end, :, end]) .* 0.5f0

    projected_ts = reshape(n.projector(reshape(posterior_latent, latent_dimensions, batch_size * time_steps), ps.projector, st.projector)[1], 1, batch_size, time_steps)

    likelihoods_time = loglike(tsmatrix, ChainRulesCore.ignore_derivatives(() -> fill(likelihood_scale, size(tsmatrix)) |> Lux.gpu), projected_ts[:, :, ts_indices])
    likelihoods = sum(likelihoods_time, dims=3)

    return posterior_latent, projected_ts, kl_divergence_time, kl_divergence, likelihoods
end

function loss(n::LatentNeuralSDE, timeseries, ps, st, beta; kwargs...)
    posterior, projected_ts, logterm, kl_divergence, distance = n(timeseries, ps, st; kwargs...)
    return -distance .+ (beta .* kl_divergence)
end
