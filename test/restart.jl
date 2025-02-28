@testset "Test restart with a different initial guess: $fun" for (fun, s) in (
  (:R2N, :R2NSolver),
  (:R2N_exact, :R2NSolver),
  (:R2N_CR, :R2NSolver),
  (:R2N_MINRES, :R2NSolver),
  (:R2N_CG_LSR1, :R2NSolver),
  (:R2, :FoSolver),
  (:fomo, :FomoSolver),
  (:lbfgs, :LBFGSSolver),
  (:tron, :TronSolver),
  (:trunk, :TrunkSolver),
)
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])
  if fun == :R2N_exact
    nlp = LBFGSModel(nlp)
    solver = eval(s)(nlp,subsolver_type = JSOSolvers.ShiftedLBFGSSolver)
  elseif fun == :R2N_CR
    solver = eval(s)(nlp,subsolver_type = CrSolver)
  elseif fun == :R2N_MINRES
    solver = eval(s)(nlp,subsolver_type = MinresSolver)    
  elseif fun == :R2N_CG_LSR1
    solver = eval(s)(LSR1Model(nlp))
  else 
    solver = eval(s)(nlp)
  end

  stats = GenericExecutionStats(nlp)

  stats = SolverCore.solve!(solver, nlp, stats)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  nlp.meta.x0 .= 2.0
  SolverCore.reset!(solver)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
end

@testset "Test restart NLS with a different initial guess: $fun" for (fun, s) in (
  (:tron, :TronSolverNLS),
  (:trunk, :TrunkSolverNLS),
)
  F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
  nlp = ADNLSModel(F, [-1.2; 1.0], 2)

  stats = GenericExecutionStats(nlp)
  solver = eval(s)(nlp)
  stats = SolverCore.solve!(solver, nlp, stats)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  nlp.meta.x0 .= 2.0
  SolverCore.reset!(solver)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)
end

@testset "Test restart with a different problem: $fun" for (fun, s) in (
  (:R2N, :R2NSolver),
  (:R2N_exact, :R2NSolver),
  (:R2N_CR, :R2NSolver),
  (:R2N_MINRES, :R2NSolver),
  (:R2N_CG_LSR1, :R2NSolver),
  (:R2, :FoSolver),
  (:fomo, :FomoSolver),
  (:lbfgs, :LBFGSSolver),
  (:tron, :TronSolver),
  (:trunk, :TrunkSolver),
)
  f(x) = (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f, [-1.2; 1.0])
  if fun == :R2N_exact
    nlp = LBFGSModel(nlp)
    solver = eval(s)(nlp,subsolver_type = JSOSolvers.ShiftedLBFGSSolver)
  elseif fun == :R2N_CR
    solver = eval(s)(nlp,subsolver_type = CrSolver)
  elseif fun == :R2N_MINRES
    solver = eval(s)(nlp,subsolver_type = MinresSolver)
  elseif fun == :R2N_CG_LSR1
    solver = eval(s)(LSR1Model(nlp))
  else 
    solver = eval(s)(nlp) 
  end

  stats = GenericExecutionStats(nlp)
  stats = SolverCore.solve!(solver, nlp, stats)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  f2(x) = (x[1])^2 + 4 * (x[2] - x[1]^2)^2
  nlp = ADNLPModel(f2, [-1.2; 1.0])
  if fun == :R2N_exact
    nlp = LBFGSModel(nlp)
  else 
    solver = eval(s)(nlp) 
  end
  SolverCore.reset!(solver, nlp)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [0.0; 0.0], atol = 1e-6)
end

@testset "Test restart NLS with a different problem: $fun" for (fun, s) in (
  (:tron, :TronSolverNLS),
  (:trunk, :TrunkSolverNLS),
  (:R2NSolverNLS, :R2NSolverNLS),
  (:R2NSolverNLS_CG, :R2NSolverNLS),
  (:R2NSolverNLS_LSQR, :R2NSolverNLS),
  (:R2NSolverNLS_CR, :R2NSolverNLS),
  (:R2NSolverNLS_LSMR, :R2NSolverNLS),
  (:R2NSolverNLS_QRMumps, :R2NSolverNLS)
)
  F(x) = [x[1] - 1; 2 * (x[2] - x[1]^2)]
  nlp = ADNLSModel(F, [-1.2; 1.0], 2)

  stats = GenericExecutionStats(nlp)
  if name == :R2NSolverNLS_CG
    solver = eval(s)(nlp, subsolver_type = CGSolver)
  elseif name == :R2NSolverNLS_LSQR
    solver = eval(s)(nlp, subsolver_type = LSQRSolver)
  elseif name == :R2NSolverNLS_CR
    solver = eval(s)(nlp, subsolver_type = CrSolver)
  elseif name == :R2NSolverNLS_LSMR
    solver = eval(s)(nlp, subsolver_type = LSMRSolver)
  elseif name == :R2NSolverNLS_QRMumps
    solver = eval(s)(nlp, subsolver_type = QRMumpsSolver)
  else
    solver = eval(s)(nlp)
  end
  stats = SolverCore.solve!(solver, nlp, stats)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [1.0; 1.0], atol = 1e-6)

  F2(x) = [x[1]; 2 * (x[2] - x[1]^2)]
  nlp = ADNLSModel(F2, [-1.2; 1.0], 2)
  SolverCore.reset!(solver, nlp)

  stats = SolverCore.solve!(solver, nlp, stats, atol = 1e-10, rtol = 1e-10)
  @test stats.status == :first_order
  @test isapprox(stats.solution, [0.0; 0.0], atol = 1e-6)
end
