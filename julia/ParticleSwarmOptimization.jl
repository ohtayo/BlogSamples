# ------------------
# # Juliaを使ったParticle Swarm Optimization(PSO)アルゴリズムの実装
# 
# - Rosenbrock functionの最適化
# - 多目的最適化にも拡張可能なように解および問題を定義
# ------------------

# Definition of a continuous optimization problem-------------------------
# 連続値最適化問題----------------------------------------------------------
"""
Abstract type of optimization problem.
最適化問題の抽象型
"""
abstract type AbstractProblem{T <: Number} end

"""
Concrete type of continuous optimization problem .
連続値最適化問題の具体型

# Fields
`objfunc`: julia function object of the objective function, 目的関数のオブジェクト
`numobjs`: number of objectives, 目的数
`numvars`: number of variables, 変数の数
`confunc`: julia function object of the constraint function, 制約計算する関数のオブジェクト
`numcons`: number of constraints, 制約の数
`limit`: upper and lower limit of the objective values, 目的関数の上下限値
`bound`: upper and lower bound of the variable values, 変数の上下限値
"""
mutable struct DoubleProblem{T} <: AbstractProblem{T}
    objfunc
    numobjs
    numvars
    confunc
    numcons
    limit::Matrix{Float64}
    bound::Matrix{T}
end
function DoubleProblem(objfunc; numvars=1, numobjs=1, confunc=NaN, numcons=0, limit=fill(NaN, 1, 1), bound=repeat([0.0 1.0], numvars))
    DoubleProblem(objfunc, numobjs, numvars, confunc, numcons, limit, bound)
end

"""
    norm(value, min, max)

Normalize value to 0-1.
`value`の値を0-1の範囲に正規化する
"""
norm(value, min, max) = (value - min) ./ (max - min)

"""
    denorm(value, min, max)

Denormalize value from 0-1 to min-max.
0-1に正規化された`value`の値をmin-maxの範囲に非正規化する
"""
denorm(value, min, max) = value .* (max - min) + min

"""
    evaluate!(problem, solutions)

Evaluate the objective value of each solution.
各解候補を評価し目的関数値を計算する
"""
function evaluate!(problem::DoubleProblem, solutions::Vector{<:AbstractSolution})
    for s in solutions
        vars = denorm(s.variables, problem.bound[:,1], problem.bound[:,2])
        objs = problem.objfunc(vars)
        if (isnan(problem.limit[1,1]))
            s.objectives = objs
        else
            s.objectives = norm(objs, problem.limit[:,1], problem.limit[:,2])
        end
    end
end

# Algorithm definition of the Particle Swarm Optimization---------------------
# Particle Swarm Optimizationの解とアルゴリズム定義-----------------------------
"""
Abstract type of solution.
最適化問題の解の抽象型
"""
abstract type AbstractSolution{T <: Number} end
"""
Concrete type of particle in PSO algorithm.
粒子群最適化における粒子(解)の具体型
"""
mutable struct Particle{T} <: AbstractSolution{T}
    objectives::Vector{Float64}
    variables::Vector{T}
    velocities::Vector{T}
    pbestvars::Vector{T}
    pbestobjs::Vector{Float64}
    constraints::Vector{T}
end

"""
    Particle(numvars[, numobjs, numcons]) -> Particle

Outer constractor of `Particle`.
`Particle`型の外部コンストラクタ

# Arguments 引数
- `numvars`: dimention (number of variables), 次元(変数の数)
- `numobjs = 1`: number of objectives, 目的数
- `numcons = 0`: number of constraints, 制約数
"""
function Particle(numvars, numobjs=1, numcons=0)
    objectives = ones(numobjs) .* Inf
    variables  = rand(numvars)
    velocities = zeros(numvars)
    pbestvars  = copy(variables)
    pbestobjs = ones(numobjs) .* Inf
    constraints = zeros(numcons)
    return Particle(objectives, variables, velocities, pbestvars, pbestobjs, constraints)
end

"""
    updateswarm!(swarm, gbest)

Update the position and velocity of the particles using the global best.
その世代の粒子群のうち最も良い解(gbest)を用いて，粒子の位置と速度を更新する
"""
function updateswarm!(swarm::Vector{Particle{Float64}}, gbest::Particle, c1=NaN, c2=NaN, w=NaN, vecr=false)
    for p in swarm
        if(vecr)
            r1 = rand(length(p.variables))
            r2 = rand(length(p.variables))
        else
            r1 = rand()
            r2 = rand()
        end
        if(isnan(c1)) c1 = rand() * 0.5 + 1.5 end
        if(isnan(c2)) c2 = rand() * 0.5 + 1.5 end
        if(isnan(w))  w = rand() * 0.4 + 0.1 end
        
        # 位置と速度の更新
        p.velocities = w * p.velocities + c1 * r1 * (p.pbestvars - p.variables) + c2 * r2 * (gbest.variables - p.variables)
        p.variables += p.velocities

        # 境界チェック
        for i = 1:length(p.variables)
            if (p.variables[i] > 1.0)
                p.variables[i] = 1.0
                p.velocities[i] *= -1.0
            elseif (p.variables[i] < 0.0)
                p.variables[i] = 0.0
                p.velocities[i] *= -1.0
            end
        end
    end
end

"""
    perturbation!(swarm)

Apply a mutation to each particles.
粒子位置に突然変異を適用する
"""
function perturbation!(swarm::Vector{Particle{Float64}})

end

"""
    updatepbest!(swarm)

Update the personal best of each particles.
その粒子がこれまで通った最も良い位置(pbest)を更新する
"""
function updatepbest!(swarm::Vector{Particle{Float64}})
    for p in swarm
        if (p.objectives < p.pbestobjs)
            p.pbestvars = copy(p.variables)
            p.pbestobjs = copy(p.objectives)
        end
    end
end

"""
    updategbest!(swarm, gbest)

Update the global best particle.
その世代の粒子群のうち最も良い解(gbest)を更新する
"""
function updategbest!(swarm::Vector{Particle{Float64}}, gbest::Particle{Float64})
    for p in swarm
        if (p.objectives < gbest.objectives)
            gbest.variables = copy(p.variables)
            gbest.objectives = copy(p.objectives)
        end
    end
end

"""
    pso(problem, iter) -> Particle

Algorithm of the Particle Swarm Optimization.
粒子群最適化アルゴリズムを実行しgbestを返す

# Arguments 引数
- `problem`: 問題の型(DoubleProblem型)
- `iter`: iterations, 繰り返し回数
- `nump`: number of particles, 粒子群サイズ．指定しなければ変数の数×10
- `c1`,`c2`,`w`: the parameters of PSO.
- `vecr`: use vectorized `r1`, `r2`. `r1`, `r2`をベクトル化するか否か(t/f)
"""
function pso(problem::DoubleProblem; iter, nump=problem.numvars*10, c1=NaN, c2=NaN, w=NaN, vecr=false)
    # 粒子群の初期化
    swarm = [Particle(problem.numvars) for i = 1:nump]
    gbest = Particle(problem.numvars)
    evaluate!(problem, swarm)
    updategbest!(swarm, gbest)
    updatepbest!(swarm)

    # 最適化の実行
    while (iter > 0)
        updateswarm!(swarm, gbest, c1, c2, w, vecr)
        perturbation!(swarm)
        evaluate!(problem, swarm)
        updategbest!(swarm, gbest)
        updatepbest!(swarm)
        iter -= 1
    end
    return gbest
end

"""
    rosenbrock(x) -> Float64

Function formulation of the Rosenbrock function.
See also: [Rosenbrock function - Wikipedia](https://en.wikipedia.org/wiki/Rosenbrock_function)
ローゼンブロック関数の定義
"""
function rosenbrock(x)
    [sum([ 100(x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i in 1:length(x) - 1])]
end

"""
Main function.
"""
function main()
    # 最適化問題の定義
    numvars = 3
    bound = [[-5.0 5.0];[-5.0 5.0];[-5.0 5.0]]
    problem = DoubleProblem(rosenbrock, numvars=numvars, bound=bound)

    # 最適化の実行
    res = pso(problem, iter=1000, nump=100)

    # 結果の表示
    vars = denorm(res.variables, bound[:,1], bound[:,2])
    objs = res.objectives
    println("objectives: " * string(objs))
    println("variables: " * string(vars))
end

@time main()

