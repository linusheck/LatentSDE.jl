using Lux
using DifferentialEquations
using Functors
using Plots
using ComponentArrays
using Zygote
using Random
using FiniteDiff
using SciMLSensitivity
using DiffEqNoiseProcess
using Random123
using JLD2

@testset "Latent SDE" begin
    solver = EulerHeun()
    tspan = (0f0, 5f0)
    datasize = 50
    
    rng = Xoshiro()

    latent_sde = StandardLatentSDE(solver, tspan, datasize, prior_size=2, posterior_size=2, diffusion_size=2)
    ps_, st = Lux.setup(rng, latent_sde)
    ps = ComponentArray{Float32}(ps_)

    input1 = (t=collect(range(tspan[1],tspan[end],datasize)),u=collect(range(0f0, 1f0, datasize)))
    input2 = (t=collect(range(tspan[1],tspan[end],datasize)),u=collect(range(2f0, 1f0, datasize)))
    
    seed = rand(rng, UInt32)

    posterior_latent, posterior_data, logterm_, kl_divergence_, distance_ = latent_sde(Timeseries([input1, input2, input2, input2]), ps, st, seed=seed)
    
    # p = plot(posterior_latent)
    # savefig(p, "posterior_latent.pdf")
    # 
    sense = BacksolveAdjoint(autojacvec=ZygoteVJP(), checkpointing=false)
    noise = function(seed, noise_size)
        rng_tree = Xoshiro(seed)
        VirtualBrownianTree(-5f0, fill(0f0, noise_size), tend=tspan[end]+5f0; rng=Threefry4x((rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32), rand(rng_tree, UInt32))))
    end
    
    input = Timeseries([input1, input1, input2, input2])
    function loss(ps)
        sum(NeuralSDEExploration.loss(latent_sde, input, ps, st, 1f0; seed=seed, sense=sense, noise=noise))
    end
    
    @test loss(ps) == loss(ps)
    
    grads_zygote_1 = Zygote.gradient(loss, ps)[1]
    @time grads_zygote_2 = Zygote.gradient(loss, ps)[1]
    grads_finitediff_1 = FiniteDiff.finite_difference_gradient(loss, ps) 
    @time grads_finitediff_2 = FiniteDiff.finite_difference_gradient(loss, ps) 

    @test grads_zygote_1 == grads_zygote_2
    @test grads_finitediff_1 == grads_finitediff_2
    outliers = findall(x -> !isapprox(x[1], x[2], rtol=1.0), collect(zip(vec(grads_zygote_1), vec(grads_finitediff_1))))
    
    println(outliers)
    println(grads_zygote_1[outliers])
    println(grads_finitediff_1[outliers])
    println(labels(grads_zygote_1)[outliers])
    println(labels(grads_finitediff_1)[outliers])
    println("Please check the grads manually:")
    println(grads_zygote_1)
    println(grads_finitediff_1)
    @test length(outliers) < length(ps) / 10
end
