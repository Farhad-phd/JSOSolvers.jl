
using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools
using LinearAlgebra

# this package
using Revise
using JSOSolvers

# F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
# nlp = ADNLSModel(F, [-1.2; 1.0], 2)

# stats_R2NLS= R2NLS(nlp,verbose=1,η1= 0.0001 ,η2=0.001, λ=2, max_iter = 1200)


# myTrunkSolver= TrunkSolverNLS(nlp)
# stats_trunkls =   GenericExecutionStats(nlp)
# SolverCore.solve!(myTrunkSolver, nlp, stats_trunkls; verbose=1, max_iter = 10)



for (name, s) in (
#   (:trunk, :TrunkSolverNLS),
  (:R2NLSSolver, :R2NLSSolver),
)
  F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
  nlp = ADNLSModel(F, [-1.2; 1.0], 2)

  stats = GenericExecutionStats(nlp)
  
  solver = eval(s)(nlp)
  
  stats = SolverCore.solve!(solver, nlp, stats,verbose=1 ,max_iter = 100)
  

  F2(x) = [x[1]; 2 * (x[2] - x[1]^2)]
  nlp = ADNLSModel(F2, [-1.2; 1.0], 2)
  SolverCore.reset!(solver, nlp)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10, verbose=1, max_iter = 100,)
  print(stats)
end