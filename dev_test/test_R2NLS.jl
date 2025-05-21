using ADNLPModels, Krylov, LinearOperators, NLPModels, NLPModelsModifiers, SolverCore, SolverTools
using LinearAlgebra

# this package
using Revise
using JSOSolvers



for (name, s) in (
  (:trunk, :TrunkSolverNLS),
  (:R2NLSSolver, :R2NLSSolver),
)
  println("Testing $name")
  F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
  nlp = ADNLSModel(F, [-1.2; 1.0], 2)

  stats = GenericExecutionStats(nlp)
  
  solver = eval(s)(nlp)
  
  stats = SolverCore.solve!(solver, nlp, stats, verbose=1 ,max_iter = 100)
  

#   F2(x) = [x[1]; 2 * (x[2] - x[1]^2)]
#   nlp = ADNLSModel(F2, [-1.2; 1.0], 2)
#   SolverCore.reset!(solver, nlp)

#   stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10, verbose=1, max_iter = 100,)
#   print(stats)
end