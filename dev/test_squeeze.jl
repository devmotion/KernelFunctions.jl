using KernelFunctions; const KF = KernelFunctions
using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using StaticArrays
using ForwardDiff
using Zygote
using GradDescent
k = SqExponentialKernel(4.0)+2*(Matern32Kernel([5.0,3.0])*SqExponentialKernel(FunctionTransform(x->sin.(x))))+3.0*LinearKernel(1.0,3.0)

t = KernelFunctions.params(k)

# tree = Tree(t)
#
# for it in PreOrderDFS(tree)
#     @info "Children" AbstractTrees.has_children(it)
#     @info it
# end

##
# v_flatten(t::AbstractVector{<:Real}) = Iterators.flatten(t)
# function v_flatten(t::AbstractVector)
#     vcat(v_flatten.(t)...)
# end
#
# function v_flatten(t::Tuple)
#     vcat(v_flatten.(t)...)
# end
X = randn(100,5)
y = rand(100)
ks = SqExponentialKernel([2.0,3.0,4.0,5.0,6.0])
p = KernelFunctions.opt_params(ks)
function fff(x)
    @show x
    ps[i] = x;
    kernelmatrix(KernelFunctions.duplicate(kernel,ps),X,obsdim=1);
end

# [kernelderivative(kernel,ps,ps[i],i,X) for i in 1:length(ps)]
[]
tail(v::Vector) = view(v,2:length(v))
KF.duplicate(transform(ks),first(p))
ksum = 0.1*ks + 0.3*ks
kprod = ks*ks
KF.params(k)
AGP.kernelderivative(ks,X)
kts = SqExponentialKernel(ScaleTransform(2.0)∘SelectTransform([2,3,4]))
KF.params(kts)

K = ksum
kw = AGP.wrapper(K,Adam())
m = VGP(X,y,K,StudentTLikelihood(3.0),AnalyticVI())
train!(m,2)
gp = m.f[1]
f1,_,_ = AGP.hyperparameter_gradient_function(gp,m.X)
Js = AGP.kernelderivative(kw,X)
grads = AGP.compute_hyperparameter_gradient(kw,f1,Js)
AGP.apply_gradients_lengthscale!(kw,grads)
using ForwardDiff
using Zygote
KernelFunctions.duplicate(k,t)
ForwardDiff.jacobian(x->kernelmatrix(duplicate(k,x),X,obsdim=2),t)
