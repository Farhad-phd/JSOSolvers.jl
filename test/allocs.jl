"""
    @wrappedallocs(expr)
Given an expression, this macro wraps that expression inside a new function
which will evaluate that expression and measure the amount of memory allocated
by the expression. Wrapping the expression in a new function allows for more
accurate memory allocation detection when using global variables (e.g. when
at the REPL).
For example, `@wrappedallocs(x + y)` produces:
```julia
function g(x1, x2)
    @allocated x1 + x2
end
g(x, y)
```
You can use this macro in a unit test to verify that a function does not
allocate:
```
@test @wrappedallocs(x + y) == 0
```
"""
macro wrappedallocs(expr)
  argnames = [gensym() for a in expr.args]
  quote
    function g($(argnames...))
      @allocated $(Expr(expr.head, argnames...))
    end
    $(Expr(:call, :g, [esc(a) for a in expr.args]...))
  end
end

if Sys.isunix()
  @testset "Allocation tests" begin
    @testset "$name" for (name, symsolver) in (
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
      for model in NLPModelsTest.nlp_problems
        nlp = eval(Meta.parse(model))()
        if unconstrained(nlp) || (bound_constrained(nlp) && (name == :TronSolver))
          if (name == :FoSolver || name == :FomoSolver)
            solver = eval(symsolver)(nlp; M = 2) # nonmonotone configuration allocates extra memory
          elseif name == :R2N_exact
            solver =
              eval(symsolver)(LBFGSModel(nlp), subsolver_type = JSOSolvers.ShiftedLBFGSSolver)
          elseif name == :R2N_CR
            solver = eval(symsolver)(nlp, subsolver_type = CrSolver)
          elseif name == :R2N_MINRES
            solver = eval(symsolver)(nlp, subsolver_type = MinresSolver)
          elseif name == :R2N_CG_LSR1
            solver = eval(symsolver)(LSR1Model(nlp))
          else
            solver = eval(symsolver)(nlp)
          end
          if name == :FomoSolver
            T = eltype(nlp.meta.x0)
            stats = GenericExecutionStats(nlp, solver_specific = Dict(:avgβmax => T(0)))
          else
            stats = GenericExecutionStats(nlp)
          end
          with_logger(NullLogger()) do
            SolverCore.solve!(solver, nlp, stats)
            reset!(solver)
            reset!(nlp)
            al = @wrappedallocs SolverCore.solve!(solver, nlp, stats)
            @test al == 0
          end
        end
      end
    end

    @testset "$name" for (name, symsolver) in (
      (:TrunkSolverNLS, :TrunkSolverNLS),
      (:R2NSolverNLS, :R2NSolverNLS),
      (:R2NSolverNLS_CG, :R2NSolverNLS),
      (:R2NSolverNLS_LSQR, :R2NSolverNLS),
      (:R2NSolverNLS_CR, :R2NSolverNLS),
      (:R2NSolverNLS_LSMR, :R2NSolverNLS),
      (:R2NSolverNLS_QRMumps, :R2NSolverNLS),
      (:TronSolverNLS, :TronSolverNLS),
    )
      for model in NLPModelsTest.nls_problems
        nlp = eval(Meta.parse(model))()
        if unconstrained(nlp) || (bound_constrained(nlp) && (symsolver == :TronSolverNLS))
          if name == :R2NSolverNLS_CG
            solver = eval(symsolver)(nlp, subsolver_type = CGSolver)
          elseif name == :R2NSolverNLS_LSQR
            solver = eval(symsolver)(nlp, subsolver_type = LSQRSolver)
          elseif name == :R2NSolverNLS_CR
            solver = eval(symsolver)(nlp, subsolver_type = CrSolver)
          elseif name == :R2NSolverNLS_LSMR
            solver = eval(symsolver)(nlp, subsolver_type = LSMRSolver)
          elseif name == :R2NSolverNLS_QRMumps
            solver = eval(symsolver)(nlp, subsolver_type = QRMumpsSolver)
          else
            solver = eval(symsolver)(nlp)
          end
          stats = GenericExecutionStats(nlp)
          with_logger(NullLogger()) do
            SolverCore.solve!(solver, nlp, stats)
            reset!(solver)
            reset!(nlp)
            al = @wrappedallocs SolverCore.solve!(solver, nlp, stats)
            @test al == 0
          end
        end
      end
    end
  end
end
