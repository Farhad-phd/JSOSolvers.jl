
# additional packages
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools

# this package
using Revise
using JSOSolvers


T = Float64
x0 = [T(0)]
f(x) = -exp(x[1])
nlp = ADNLPModel(f, x0)



# println("======================================================================")

# stats3 = R2N(nlp, subsolver_type= CgLanczosShiftSolver ,verbose=1, max_iter=10)
# println("CgLanczosShiftSolver _ ADNLPModel status = ", stats3)


# f(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
# nlp = ADNLSModel(f, [-1.2; 1.0], 2)

println("======================================================================")

stats = R2N(nlp, verbose=1, max_iter=10,subsolver_verbose=0 )
println("stats = ", stats)
println("======================================================================")


stats_LBFGS= R2N(LBFGSModel(nlp), verbose=1, max_iter=10,subsolver_verbose=0)
println("stats_LBFGS = ", stats_LBFGS)

println("======================================================================")

stats_exact = R2N(LBFGSModel(nlp), subsolver_type = JSOSolvers.ShiftedLBFGSSolver, verbose=1, max_iter=10)
println("stats_exact = ", stats_exact)



println("======================================================================")
stats2 = R2N(LBFGSModel(nlp), subsolver_type=CgSolver ,verbose=1, max_iter=10)
println("CG _ LBFGSModel status = ", stats2)


println("======================================================================")
stats3 = R2N(nlp, subsolver_type=CgSolver ,verbose=1, max_iter=10)
println("CG _ ADNLPModel status = ", stats3)



println("======================================================================")
stats4 = R2N(nlp, subsolver_type= CrSolver ,verbose=100, max_iter=1000)
println("CR _ ADNLPModel status = ", stats4)