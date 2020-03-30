using KernelFunctions
using KernelFunctions: metric
using Distances

using LinearAlgebra
using Random
using Test

rng = MersenneTwister(123456)
x = rand(rng) * 2
v1 = rand(rng, 3)
v2 = rand(rng, 3)
@testset "RationalQuadraticKernel" begin
    α = 2.0
    k = RationalQuadraticKernel(α = α)
    @test RationalQuadraticKernel(alpha = α).α == [α]
    @test kappa(k, x) ≈ (1.0 + x / 2.0)^-2
    @test k(v1, v2) ≈ (1.0 + norm(v1 - v2)^2 / 2.0)^-2
    @test kappa(RationalQuadraticKernel(α = α), x) == kappa(k, x)
    @test metric(RationalQuadraticKernel()) == SqEuclidean()
    @test metric(RationalQuadraticKernel(α = 2.0)) == SqEuclidean()
end
@testset "GammaRationalQuadraticKernel" begin
    k = GammaRationalQuadraticKernel()
    @test kappa(k, x) ≈ (1.0 + x^2.0 / 2.0)^-2
    @test k(v1, v2) ≈ (1.0 + norm(v1 - v2)^4.0 / 2.0)^-2
    @test kappa(GammaRationalQuadraticKernel(), x) == kappa(k, x)
    a = 1.0 + rand()
    @test GammaRationalQuadraticKernel(alpha = a).α == [a]
        # Coherence test
    @test kappa(GammaRationalQuadraticKernel(α = a, γ = 1.0), x) ≈ kappa(RationalQuadraticKernel(α = a), x)
    @test metric(GammaRationalQuadraticKernel()) == SqEuclidean()
    @test metric(GammaRationalQuadraticKernel(γ = 2.0)) == SqEuclidean()
    @test metric(GammaRationalQuadraticKernel(γ = 2.0, α = 3.0)) == SqEuclidean()
end
