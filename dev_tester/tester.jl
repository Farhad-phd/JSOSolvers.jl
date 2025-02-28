# additional packages
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools
using LinearAlgebra

# this package
using Revise
using JSOSolvers


using QuadraticModels, RipQP, SparseArrays


"""
    kkt_checker(nlp, sol; kwargs...)

Given an NLPModels `nlp` and a vector `sol`, it returns the KKT residual of an optimization problem as a tuple (primal, dual).
In particular, it uses `ripqp` to solve the following quadratic optimization problem with linear constraints
```
min_{d} ∇f(sol)ᵀd +  ½∥d∥²
        lvar ≤ sol + d ≤ uvar
        lcon ≤ c(sol) + ∇c(sol)d ≤ ucon
```
The solution of this problem is the gradient of the Lagrangian of the `nlp` at `sol` thanks to the ½ ‖d‖² term in the objective.

Keyword arguments are passed to `RipQP`.
"""
function kkt_checker(
  nlp::AbstractNLPModel{T, S},
  sol;
  kwargs...,
) where {T, S}
  nvar = nlp.meta.nvar
  g = grad(nlp, sol)
  Hrows, Hcols = collect(1:nvar), collect(1:nvar)
  Hvals = ones(T, nlp.meta.nvar)

  feas_res = max.(nlp.meta.lvar - sol, sol - nlp.meta.uvar, 0)
  kkt_nlp = if nlp.meta.ncon > 0
    c = cons(nlp, sol)
    feas_res = vcat(max.(nlp.meta.lcon - c, c - nlp.meta.ucon, 0), feas_res)
    Arows, Acols = jac_structure(nlp)
    Avals = jac_coord(nlp, sol)
    QuadraticModel(
      g,
      Hrows,
      Hcols,
      Hvals,
      Arows = Arows,
      Acols = Acols,
      Avals = Avals,
      lcon = nlp.meta.lcon .- c,
      ucon = nlp.meta.ucon .- c,
      lvar = nlp.meta.lvar .- sol,
      uvar = nlp.meta.uvar .- sol,
      x0 = fill!(S(undef, nlp.meta.nvar), zero(T)),
    )
  else
    QuadraticModel(
      g,
      Hrows,
      Hcols,
      Hvals,
      lvar = nlp.meta.lvar .- sol,
      uvar = nlp.meta.uvar .- sol,
      x0 = fill!(S(undef, nlp.meta.nvar), zero(T)),
    )
  end
  stats = ripqp(kkt_nlp; display = false, sp = K2LDLParams{T}(ρ0 = T(1.0e-2), δ0 = T(1.0e-2)), kwargs...)
  if !(stats.status ∈ (:acceptable, :first_order))
    @warn "Failure in the Lagrange multiplier computation, the status of ripqp is $(stats.status)."
  end
  dual_res = stats.solution
  return feas_res, dual_res
end


# T=Float64
T= Float16

f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
x0 = T[-1.2; 1.0]

nlp =  ADNLPModel(f, x0)

ϵ = eps(T)^T(1 / 4)

ng0 = norm(grad(nlp, nlp.meta.x0))

stats= R2N(nlp, atol = ϵ, rtol = ϵ, verbose =1, max_iter=20)
println("====================================")
stats_CR= R2N(nlp, atol = ϵ, rtol = ϵ, verbose =1, subsolver_type = CrlsSolver, max_iter=20, subsolver_verbose=0)


primal, dual = kkt_checker(nlp, stats.solution)

println(all(abs.(dual) .< ϵ * ng0 + ϵ))
println(all(abs.(primal) .< ϵ * ng0 + ϵ))
println(stats.dual_feas < ϵ * ng0 + ϵ) 


primal, dual = kkt_checker(nlp, stats_CR.solution)

println(all(abs.(dual) .< ϵ * ng0 + ϵ))
println(all(abs.(primal) .< ϵ * ng0 + ϵ))
println(stats_CR.dual_feas < ϵ * ng0 + ϵ)


# test CR  and CG 
# Symmetric and positive definite systems.
T = Float16
function symmetric_definite(n :: Int=10; FC=Float64)
    α = FC <: Complex ? FC(im) : one(FC)
    A = spdiagm(-1 => α * ones(FC, n-1), 0 => 4 * ones(FC, n), 1 => conj(α) * ones(FC, n-1))
    b = A * FC[1:n;]
    return A, b
  end

A, b = symmetric_definite(FC=T)
subtol = max(rtol, min(T(0.1), √T(1.3e+01), T(0.9) * one(T)))

(x, stats) = cr(A, b,atol = ϵ, rtol = subtol,verbose=1,linesearch=true)
println(x, stats)

(x,stats_cg) = cg(A, b, atol = ϵ, rtol = subtol, verbose=1, linesearch=true)
println(x, stats_cg)

#############################################################

# Test CR and CG with Float16
T = Float16
ϵ = eps(T)^T(1 / 4)
rtol = ϵ 
function symmetric_definite(n::Int = 10; FC = Float64)
  α = FC <: Complex ? FC(im) : one(FC)
  A = spdiagm(-1 => α * ones(FC, n-1), 0 => 4 * ones(FC, n), 1 => conj(α) * ones(FC, n-1))
  b = A * FC[1:n;]
  return A, b
end

# Create a symmetric, positive definite system using Float16
A, b = symmetric_definite(FC = T)

# Define tolerances (ensure that rtol and ϵ are defined appropriately for Float16)
subtol = max(rtol, min(T(0.1), √T(1.3e+01), T(0.9) * one(T)))

# Test the CR solver
(x, stats) = cr(A, b, atol = ϵ, rtol = subtol, verbose = 1, linesearch = true)
println("CR result:")
println(x, stats)

# Test the CG solver
(x, stats_cg) = cg(A, b, atol = ϵ, rtol = subtol, verbose = 1, linesearch = true)
println("CG result:")
println(x, stats_cg)
