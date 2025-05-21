using LinearAlgebra, Arpack, SparseArrays
using GenericLinearAlgebra

# use Arpack to obtain largest eigenvalue in magnitude with a minimum of robustness
function LinearAlgebra.opnorm(B; kwargs...)
  m, n = size(B)
  opnorm_fcn = m == n ? opnorm_eig : opnorm_svd
  return opnorm_fcn(B; kwargs...)
end

function opnorm_eig(B; max_attempts::Int = 3)
  have_eig = false
  attempt = 0
  λ = zero(eltype(B))
  n = size(B, 1)
  nev = 1
  ncv = max(20, 2 * nev + 1)

  # 1) If BigFloat, use pure-Julia dense eigenvals from GLA
  if eltype(B) === BigFloat
    println("Using GenericLinearAlgebra for BigFloat")
    F = GenericLinearAlgebra.eigen(Matrix{BigFloat}(B))
    return maximum(abs, F.values), true
  end

  # 2) If small (n ≤ 5), do a dense LAPACK eigen (fast for small Float64)
  if n ≤ 5
    F = eigen(Matrix(B))
    return maximum(abs, F.values), true
  end

  # 3) Otherwise use Arpack’s iterative eigs

  while !(have_eig || attempt >= max_attempts)
    attempt += 1
    try
      # Estimate largest eigenvalue in absolute value
      d, nconv, niter, nmult, resid =
        eigs(B; nev = nev, ncv = ncv, which = :LM, ritzvec = false, check = 1)

      # Check if eigenvalue has converged
      have_eig = nconv == 1
      if have_eig
        λ = abs(d[1])  # Take absolute value of the largest eigenvalue
        break  # Exit loop if successful
      else
        # Increase NCV for the next attempt if convergence wasn't achieved
        ncv = min(2 * ncv, n)
      end
    catch e
      if occursin("XYAUPD_Exception", string(e)) && ncv < n
        @warn "Arpack error: $e. Increasing NCV to $ncv and retrying."
        ncv = min(2 * ncv, n)  # Increase NCV but don't exceed matrix size
      else
        rethrow(e)  # Re-raise if it's a different error
      end
    end
  end

  return λ, have_eig
end

function opnorm_svd(J; max_attempts::Int = 3)
  have_svd = false
  attempt = 0
  σ = zero(eltype(J))
  n = min(size(J)...)  # Minimum dimension of the matrix
  nsv = 1
  ncv = 10

  # If BigFloat, use dense and GenericLinearAlgebra
  if eltype(J) === BigFloat
    # GenericLinearAlgebra.svd returns a SVD object with .S
    S = GenericLinearAlgebra.svd(Matrix(J))
    return maximum(S.S), true
  end

  # If small matrix just do dense
  if n ≤ 5 
    σs = LinearAlgebra.svd(Matrix(J)).S
    return maximum(σs), true
  end

  while !(have_svd || attempt >= max_attempts)
    attempt += 1
    try
      # Estimate largest singular value
      s, nconv, niter, nmult, resid = svds(J; nsv = nsv, ncv = ncv, ritzvec = false, check = 1)

      # Check if singular value has converged
      have_svd = nconv >= 1
      if have_svd
        σ = maximum(s.S)  # Take the largest singular value
        break  # Exit loop if successful
      else
        # Increase NCV for the next attempt if convergence wasn't achieved
        ncv = min(2 * ncv, n)
      end
    catch e
      if occursin("XYAUPD_Exception", string(e)) && ncv < n
        @warn "Arpack error: $e. Increasing NCV to $ncv and retrying."
        ncv = min(2 * ncv, n)  # Increase NCV but don't exceed matrix size
      else
        rethrow(e)  # Re-raise if it's a different error
      end
    end
  end

  return σ, have_svd
end
