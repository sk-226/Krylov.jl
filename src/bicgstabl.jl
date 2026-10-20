export bicgstabl, bicgstabl!

function bicgstabl end

function bicgstabl! end

def_args_bicgstabl = (:(A                    ),
                      :(b::AbstractVector{FC}))

def_optargs_bicgstabl = (:(x0::AbstractVector),)

def_kwargs_bicgstabl = (:(; c::AbstractVector{FC} = b    ),
                        :(; atol::T = √eps(T)            ),
                        :(; rtol::T = √eps(T)            ),
                        :(; itmax::Int = 0               ),
                        :(; timemax::Float64 = Inf       ),
                        :(; verbose::Int = 0             ),
                        :(; history::Bool = false        ),
                        :(; callback = workspace -> false),
                        :(; iostream::IO = kstdout       ))

def_kwargs_workspace_bicgstabl = (:(; ℓ::Int = 2),)

def_kwargs_bicgstabl = extract_parameters.(def_kwargs_bicgstabl)
def_kwargs_workspace_bicgstabl = extract_parameters.(def_kwargs_workspace_bicgstabl)

args_bicgstabl = (:A, :b)
optargs_bicgstabl = (:x0,)
kwargs_bicgstabl = (:c, :atol, :rtol, :itmax, :timemax, :verbose, :history, :callback, :iostream)
kwargs_workspace_bicgstabl = (:ℓ,)

@eval begin
  function bicgstabl!(workspace :: BicgstablWorkspace{T,FC,S}, $(def_args_bicgstabl...); $(def_kwargs_bicgstabl...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, S <: AbstractVector{FC}}
    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    m, n = size(A)
    ℓ = workspace.ℓ
    (m == workspace.m && n == workspace.n) || error("(workspace.m, workspace.n) = ($(workspace.m), $(workspace.n)) is inconsistent with size(A) = ($m, $n)")
    m == n || error("System must be square")
    length(b) == m || error("Inconsistent problem size")
    (verbose > 0) && @printf(iostream, "BiCGstab(ℓ): system of size %d\n", n)
    ℓ ≥ 1 || error("ℓ must be at least 1")

    # Check type consistency.
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-vector products."
    ktypeof(b) == S || error("ktypeof(b) must be equal to $S")
    ktypeof(c) == S || error("ktypeof(c) must be equal to $S")

    # Set up workspace.
    Δx, x, R, U = workspace.Δx, workspace.x, workspace.R, workspace.U
    τ, σ, γ, γ′, γ″ = workspace.τ, workspace.σ, workspace.γ, workspace.γ′, workspace.γ″
    warm_start = workspace.warm_start

    stats = workspace.stats
    rNorms = stats.residuals
    reset!(stats)

    if warm_start
      kmul!(R[1], A, Δx)                      # r₀ = A * Δx
      kaxpby!(n, one(FC), b, -one(FC), R[1])  # r₀ = b - r₀
    else
      kcopy!(n, R[1], b)  # r₀ = b
    end

    kfill!(x, zero(FC))
    kfill!(U[1], zero(FC))

    α = zero(FC)
    ω = one(FC)
    ρ₀ = one(FC)

    # Compute residual norm ‖r₀‖₂.
    rNorm = knorm(n, R[1])
    history && push!(rNorms, rNorm)
    if rNorm == 0
      stats.niter = 0
      stats.solved, stats.inconsistent = true, false
      stats.timer = start_time |> ktimer
      stats.status = "x is a zero-residual solution"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    iter = 0
    itmax == 0 && (itmax = 2 * n)

    ε = atol + rtol * rNorm
    (verbose > 0) && @printf(iostream, "%5s  %7s  %8s  %8s  %5s\n", "k", "‖rₖ‖", "|αₖ|", "|ωₖ|", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %8.1e  %8.1e  %.2fs\n", iter, rNorm, abs(α), abs(ω), start_time |> ktimer)

    if kdot(n, c, R[1]) == 0  # ⟨c, r₀⟩
      stats.niter = 0
      stats.solved, stats.inconsistent = false, false
      stats.timer = start_time |> ktimer
      stats.status = "Breakdown bᴴc = 0"
      warm_start && kaxpy!(n, one(FC), Δx, x)
      workspace.warm_start = false
      return workspace
    end

    # Stopping criterion.
    solved = rNorm ≤ ε
    tired = iter ≥ itmax
    breakdown = false
    breakdown_status = "unknown"
    status = "unknown"
    user_requested_exit = false
    overtimed = false

    while !(solved || tired || breakdown || user_requested_exit || overtimed)
      iter = iter + 1
      ρ₀ = - ω * ρ₀

      # Bi-CG part.
      for j in 1 : ℓ
        ρ₁ = kdot(n, c, R[j]) # ρ₁ = (c, rⱼ)
        if ρ₁ == 0
          breakdown = true
          breakdown_status = "breakdown ρ = 0"
          break
        end
        β = α * ρ₁ / ρ₀
        ρ₀ = ρ₁

        for i in 1 : j
          kaxpby!(n, one(FC), R[i], -β, U[i])  # uᵢ = rᵢ - β uᵢ
        end

        kmul!(U[j+1], A, U[j]) # uⱼ₊₁ = A * uⱼ

        α = ρ₁ / kdot(n, c, U[j+1]) # α = ρ₁ / (c, uⱼ₊₁)
        if !isfinite(α)
          breakdown = true
          breakdown_status = "breakdown (c, uⱼ₊₁) = 0"
          break
        end

        for i in 1 : j
          kaxpy!(n, -α, U[i+1], R[i])  # rᵢ = rᵢ - α uᵢ₊₁
        end

        kmul!(R[j+1], A, R[j])  # rⱼ₊₁ = A * rⱼ
        kaxpy!(n, α, U[1], x)   # x = x + α * u₁
      end
      if breakdown
        rNorm = knorm(n, R[1])
        history && push!(rNorms, rNorm)
        solved = rNorm ≤ ε
        break
      end

      # MR part (mod. Gram-Schmidt).
      pos = 0 # τᵢⱼ's position in the flattened array
      for j in 1 : ℓ
        for i in 1 : j-1
          pos += 1
          τ[pos] = kdot(n, R[i+1], R[j+1]) / σ[i]
          kaxpy!(n, -τ[pos], R[i+1], R[j+1])  # rⱼ = rⱼ - τᵢⱼ rᵢ
        end
        σ[j] = kdotr(n, R[j+1], R[j+1])
        if σ[j] == 0
          breakdown = true
          breakdown_status = "breakdown σⱼ = 0"
          break
        end
        γ′[j] = kdot(n, R[j+1], R[1]) / σ[j]
      end
      if breakdown
        rNorm = knorm(n, R[1])
        history && push!(rNorms, rNorm)
        solved = rNorm ≤ ε
        break
      end

      γ[ℓ] = γ′[ℓ]
      ω = γ[ℓ]

      # γ = T⁻¹ γ′
      for j in ℓ-1 : -1 : 1
        γ[j] = γ′[j]
        for i in j+1 : ℓ
          γ[j] = γ[j] - τ[div((i-1)*(i-2), 2) + j] * γ[i] # γⱼ -= τⱼᵢ γᵢ
        end
      end

      # γ″ = T S γ
      for j in 1 : ℓ-1
        γ″[j] = γ[j+1]
        for i in j+1 : ℓ-1
          γ″[j] = γ″[j] + τ[div((i-1)*(i-2), 2) + j] * γ[i+1] # γ″ⱼ += τⱼᵢ γᵢ₊₁
        end
      end

      kaxpy!(n, γ[1], R[1], x)
      kaxpy!(n, -γ′[ℓ], R[ℓ+1], R[1])
      kaxpy!(n, -γ[ℓ], U[ℓ+1], U[1])

      for j in 1 : ℓ - 1
        kaxpy!(n, -γ[j], U[j+1], U[1])
        kaxpy!(n, γ″[j], R[j+1], x)
        kaxpy!(n, -γ′[j], R[j+1], R[1])
      end

      # Compute residual norm ‖rₖ‖.
      rNorm = knorm(n, R[1])
      history && push!(rNorms, rNorm)

      # Update stopping criterion.
      user_requested_exit = callback(workspace) :: Bool
      solved = rNorm ≤ ε
      tired = iter ≥ itmax
      if ω == 0
        breakdown = true
        breakdown_status = "breakdown ω = 0"
      end
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %8.1e  %8.1e  %.2fs\n", iter, rNorm, abs(α), abs(ω), start_time |> ktimer)
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    breakdown           && (status = breakdown_status)
    solved              && (status = "solution good enough given atol and rtol")
    user_requested_exit && (status = "user-requested exit")
    overtimed           && (status = "time limit exceeded")

    warm_start && kaxpy!(n, one(FC), Δx, x)
    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = false
    stats.timer = ktimer(start_time)
    stats.status = status

    return workspace
  end
end
