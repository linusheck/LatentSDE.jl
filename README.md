# LatentSDE.jl

## ⚠️ Please do not use this until [Issue #854 of SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl/issues/854) is fixed. ⚠️

Library for Latent Neural SDEs in Julia. This same code, along with notebooks, tests, analysis, etc., can be found in [this companion repository](https://github.com/glatteis/NeuralSDEExploration).

[![Build Status](https://github.com/glatteis/LatentSDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/glatteis/LatentSDE.jl/actions/workflows/CI.yml?query=branch%3Amain)

For many use cases, the user can call the `StandardLatentSDE`
constructor to avoid creating every network manually. Here, we are
constructing a latent neural SDE with a two-dimensional latent space,
where the training will happen with an Euler-Heun solver with timespan
from zero to one with 20 data points:

    LatentNeuralSDE(
        initial_prior = Dense(1 => 4, bias=false),  # 4 parameters
        initial_posterior = Dense(16 => 4),  # 68 parameters
        drift_prior = Chain(
            layer_1 = Dense(2 => 64, tanh_fast),  # 192 parameters
            layer_2 = Dense(64 => 64, tanh_fast),  # 4_160 parameters
            layer_3 = Dense(64 => 2, tanh_fast),  # 130 parameters
        ),
        drift_posterior = Chain(
            layer_1 = Dense(18 => 64, tanh_fast),  # 1_216 parameters
            layer_2 = Dense(64 => 64, tanh_fast),  # 4_160 parameters
            layer_3 = Dense(64 => 2, tanh_fast),  # 130 parameters
        ),
        diffusion = Diagonal(
            layers = NamedTuple(
                layer_1 = Chain(
                    layer_1 = Dense(1 => 16, tanh_fast),  # 32 parameters
                    layer_2 = Dense(16 => 16, tanh_fast),  # 272 parameters
                    layer_3 = Dense(16 => 1, sigmoid_fast),  # 17 parameters
                    layer_4 = WrappedFunction(Base.Fix1{typeof(broadcast), LatentSDE.var"#35#38"}(broadcast, LatentSDE.var"#35#38"())),
                ),
                layer_2 = Chain(
                    layer_1 = Dense(1 => 16, tanh_fast),  # 32 parameters
                    layer_2 = Dense(16 => 16, tanh_fast),  # 272 parameters
                    layer_3 = Dense(16 => 1, sigmoid_fast),  # 17 parameters
                    layer_4 = WrappedFunction(Base.Fix1{typeof(broadcast), LatentSDE.var"#35#38"}(broadcast, LatentSDE.var"#35#38"())),
                ),
            ),
        ),
        encoder_recurrent = Recurrence(
            cell = GRUCell(1 => 16),        # 880 parameters, plus 1
        ),
        encoder_net = Dense(16 => 16),      # 272 parameters
        projector = Dense(2 => 1),          # 3 parameters
    )         # Total: 11_857 parameters,
              #        plus 1 states.

To replace one of these networks, just provide them as keyword argument.
There are also keyword arguments for the network sizes. Here, we are
giving the projector a $\tanh$ activation and providing a smaller hidden
size for the prior:

    julia> StandardLatentSDE(EulerHeun(), (0.0, 1.0), 20,
                 prior_size=32, projector=Lux.Dense(2 => 1, tanh))

As the package is based on `Lux.jl`, we need to initialize the Latent
SDE as a Lux model:

    julia> using Lux, Random, ComponentArrays
    julia> rng = Xoshiro()
    julia> ps_, st = Lux.setup(rng, latent_sde)
    julia> ps = ComponentArray{Float64}(ps_)

The `Lux.setup` function creates the parameters and state of the model.
We have to create a `ComponentArray` from the parameters because
sensitivity methods will only work with array-like types, but `ps_` is a
dictionary. A `ComponentArray` is a flat array of parameters with an
overlay that keeps the assignment of parameters to networks.

We can already plot a few samples from the prior.

    julia> plot(sample_prior_dataspace(latent_sde, ps, st, b=10))

To run the posterior, we need an input time series. Let's create one:

    julia> t = collect(range(0.0, 1.0, 20))
    julia> ts = Timeseries(t, [[[x] for x in t]])
    julia> plot(ts)

We plot ten separate posteriors by repeating the input time series ten
times:

    julia> repeated_ts = repeat_ts(10, ts)
    julia> _, posterior_dataspace, _, _, _ = latent_sde(repeated_ts, ps, st)
    julia> plot(Timeseries(t, posterior_dataspace))

Now we train the latent SDE on the data. This can be done with the
modular Julia approach. We choose the libraries `Optimisers.jl` for the
ADAM algorithm and `Zygote.jl` for the autodifferentiation. As the loss
function, we use the built-in loss function that uses a log-likelihood.

    julia> using Optimisers, Zygote
    julia> loss(ps) = sum(loss(latent_sde, repeat_ts(10, ts), ps, st, 0.01))
    julia> loss(ps)
    1.6216436199468267e6
    julia> function train()
        for iteration in 1:200
           l, dps = Zygote.withgradient(loss, ps)
           println("Loss: $l")
           Optimisers.update!(opt_state, ps, dps[1])
       end
    end
    julia> opt_state = Optimisers.setup(Optimisers.Adam(0.1), ps)
    julia> train()
    julia> opt_state = Optimisers.setup(Optimisers.Adam(0.001), ps)
    julia> train()

After training, let's plot prior and posterior:

    julia> loss(ps)
    -661.5151548380488
    julia> _, posterior_dataspace, _, _, _ = latent_sde(repeat_ts(10, ts), ps, st)
    julia> plot(Timeseries(t, posterior_dataspace))

    julia> plot(sample_prior_dataspace(latent_sde, ps, st, b=10))

The trained prior has a slightly broader initial condition than the
data, but the dynamics are accurate. The initial condition's variance
depends on the KL-weighing factor $\beta$.
