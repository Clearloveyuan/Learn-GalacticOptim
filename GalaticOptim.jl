{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Learn GalacticOptim and Optim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [],
   "source": [
    "using GalacticOptim, Optim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "rosenbrock (generic function with 3 methods)"
      ]
     },
     "execution_count": 120,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rosenbrock(x,p) =  (  - x[1])^2 + p[2] * (x[2] - x[1]^2)^2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2-element Vector{Float64}:\n",
       " 0.0\n",
       " 0.0"
      ]
     },
     "execution_count": 121,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x0 = zeros(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2-element Vector{Float64}:\n",
       "   1.0\n",
       " 100.0"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "p  = [1.0,100.0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[36mOptimizationProblem\u001b[0m. In-place: \u001b[36mtrue\u001b[0m\n",
       "u0: [0.0, 0.0]"
      ]
     },
     "execution_count": 123,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(rosenbrock,x0,p)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999634355313174\n",
       " 0.9999315506115275"
      ]
     },
     "execution_count": 124,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol = solve(prob,NelderMead())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [],
   "source": [
    "using BlackBoxOptim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[36mOptimizationProblem\u001b[0m. In-place: \u001b[36mtrue\u001b[0m\n",
       "u0: [0.0, 0.0]"
      ]
     },
     "execution_count": 126,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(rosenbrock, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting optimization with optimizer DiffEvoOpt{FitPopulation{Float64}, RadiusLimitedSelector, BlackBoxOptim.AdaptiveDiffEvoRandBin{3}, RandomBound{ContinuousRectSearchSpace}}\n",
      "0.00 secs, 0 evals, 0 steps\n",
      "\n",
      "Optimization stopped after 10001 steps and 0.04 seconds\n",
      "Termination reason: Max number of steps (10000) reached\n",
      "Steps per second = 256435.67\n",
      "Function evals per second = 259025.41\n",
      "Improvements/step = 0.21690\n",
      "Total function evaluations = 10102\n",
      "\n",
      "\n",
      "Best candidate found: [1.0, 1.0]\n",
      "\n",
      "Fitness: 0.000000000\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999999999996\n",
       " 0.9999999999999992"
      ]
     },
     "execution_count": 127,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol = solve(prob,BBO())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "OptimizationFunction{true, GalacticOptim.AutoForwardDiff{nothing}, typeof(rosenbrock), Nothing, Nothing, Nothing, Nothing, Nothing, Nothing}(rosenbrock, GalacticOptim.AutoForwardDiff{nothing}(), nothing, nothing, nothing, nothing, nothing, nothing)"
      ]
     },
     "execution_count": 129,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f2 = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[36mOptimizationProblem\u001b[0m. In-place: \u001b[36mtrue\u001b[0m\n",
       "u0: [0.0, 0.0]"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(f2, x0, p)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999999373614\n",
       " 0.999999999868622"
      ]
     },
     "execution_count": 131,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol = solve(prob,BFGS())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[36mOptimizationProblem\u001b[0m. In-place: \u001b[36mtrue\u001b[0m\n",
       "u0: [0.0, 0.0]"
      ]
     },
     "execution_count": 132,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(f2, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999899064821\n",
       " 0.9999999797630695"
      ]
     },
     "execution_count": 133,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol = solve(prob, Fminbox(GradientDescent()))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Rosenbrock function examples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [],
   "source": [
    "using GalacticOptim, Optim, Test, Random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2-element Vector{Float64}:\n",
       "   1.0\n",
       " 100.0"
      ]
     },
     "execution_count": 135,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rosenbrock(x, p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2\n",
    "x0 = zeros(2)\n",
    "_p  = [1.0, 100.0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "OptimizationFunction{true, GalacticOptim.AutoForwardDiff{nothing}, typeof(rosenbrock), Nothing, Nothing, Nothing, Nothing, Nothing, Nothing}(rosenbrock, GalacticOptim.AutoForwardDiff{nothing}(), nothing, nothing, nothing, nothing, nothing, nothing)"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f3 = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9339883554832432\n",
       " 0.8673154506302181"
      ]
     },
     "execution_count": 137,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l1 = rosenbrock(x0, _p)\n",
    "prob = OptimizationProblem(f3, x0, _p)\n",
    "sol = solve(prob, SimulatedAnnealing())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "================================================================================\n",
      "SAMIN results\n",
      "NO CONVERGENCE: MAXEVALS exceeded\n",
      "\n",
      "     Obj. value:           0.05281\n",
      "\n",
      "       parameter      search width\n",
      "         0.77067           0.35556 \n",
      "         0.59542           0.15000 \n",
      "================================================================================\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.7706700409293477\n",
       " 0.5954190179306745"
      ]
     },
     "execution_count": 143,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Random.seed!(1234)\n",
    "prob = OptimizationProblem(f3, x0, _p, lb=[-1.0, -1.0], ub=[0.8, 0.8])\n",
    "sol = solve(prob, SAMIN())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 144,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3_w,6)-aCMA-ES (mu_w=2.0,w_1=64%) in dimension 2 (seed=9180451354441353971, 2021-03-27T19:05:55.007)\n",
      "                                                                 \n",
      "     1        6   1.60109981e+00   9.58e-02   1.253e+00     0.109\n",
      "     2       12   1.50796511e+00   7.95e-02   1.245e+00     0.109\n",
      "     3       18   1.40892846e+00   6.89e-02   1.340e+00     0.109\n",
      "    89      534   4.00000000e-02   2.66e-04   8.403e+00     0.109\n",
      "(3_w,6)-aCMA-ES (mu_w=2.0,w_1=64%) in dimension 2 (seed=9180451354441353971, 2021-03-27T19:05:55.007)\n",
      "  termination reason: ftol = 1.0e-11 (2021-03-27T19:05:55.007)\n",
      "  lowest observed function value: 0.040000000000344295 at [0.7999999999947122, 0.6399999901161894]\n",
      "  population mean: [0.7999999999993345, 0.6400000261357587]\n",
      "\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.8900013797245124\n",
       " 0.6399999901161894"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "using CMAEvolutionStrategy\n",
    "sol = solve(prob, CMAEvolutionStrategyOpt())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 146,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "rosenbrock (generic function with 3 methods)"
      ]
     },
     "execution_count": 147,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rosenbrock(x, p=nothing) =  (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999634355313174\n",
       " 0.9999315506115275"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "l1 = rosenbrock(x0)\n",
    "prob = OptimizationProblem(rosenbrock, x0)\n",
    "sol = solve(prob, NelderMead())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 149,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9985769782748619\n",
       " 0.9971971594653897"
      ]
     },
     "execution_count": 151,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cons= (x,p) -> [x[1]^2 + x[2]^2]\n",
    "optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff();cons= cons)\n",
    "prob = OptimizationProblem(optprob, x0)\n",
    "sol = solve(prob, ADAM(0.1), maxiters = 1000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 152,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999999373614\n",
       " 0.999999999868622"
      ]
     },
     "execution_count": 153,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol = solve(prob, BFGS())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 154,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999999999994\n",
       " 0.9999999999999989"
      ]
     },
     "execution_count": 155,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol = solve(prob, Newton())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 156,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.999999999999108\n",
       " 0.9999999999981819"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol = solve(prob, Optim.KrylovTrustRegion())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 158,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999992619217\n",
       " 0.9999999985003628"
      ]
     },
     "execution_count": 159,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(optprob, x0, lcons = [-Inf], ucons = [Inf])\n",
    "sol = solve(prob, IPNewton())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999992669327\n",
       " 0.9999999985109471"
      ]
     },
     "execution_count": 161,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(optprob, x0, lcons = [-5.0], ucons = [10.0])\n",
    "sol = solve(prob, IPNewton())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 162,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999992541948\n",
       " 0.9999999984843432"
      ]
     },
     "execution_count": 163,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(optprob, x0, lcons = [-Inf], ucons = [Inf], lb = [-500.0,-500.0], ub=[50.0,50.0])\n",
    "sol = solve(prob, IPNewton()) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999992619217\n",
       " 0.9999999985003628"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "function con2_c(x,p)\n",
    "    [x[1]^2 + x[2]^2, x[2]*sin(x[1])-x[1]]\n",
    "end\n",
    "optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff();cons= con2_c)\n",
    "prob = OptimizationProblem(optprob, x0, lcons = [-Inf,-Inf], ucons = [Inf,Inf])\n",
    "sol = solve(prob, IPNewton())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.24327905408863862\n",
       " 0.05757865786675858"
      ]
     },
     "execution_count": 168,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cons_circ = (x,p) -> [x[1]^2 + x[2]^2]\n",
    "optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff();cons= cons_circ)\n",
    "prob = OptimizationProblem(optprob, x0, lcons = [-Inf], ucons = [0.25^2])\n",
    "sol = solve(prob, IPNewton())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 169,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test sqrt(cons(sol.minimizer,nothing)[1]) ≈ 0.25 rtol = 1e-6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.7396709479564985\n",
       " 0.5470724299209024"
      ]
     },
     "execution_count": 170,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "optprob = OptimizationFunction(rosenbrock, GalacticOptim.AutoZygote())\n",
    "prob = OptimizationProblem(optprob, x0)\n",
    "sol = solve(prob, ADAM(), maxiters = 1000, progress = false)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.799999998888889\n",
       " 0.6399999982096882"
      ]
     },
     "execution_count": 172,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])\n",
    "sol = solve(prob, Fminbox())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 173,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "================================================================================\n",
      "SAMIN results\n",
      "NO CONVERGENCE: MAXEVALS exceeded\n",
      "\n",
      "     Obj. value:           0.04776\n",
      "\n",
      "       parameter      search width\n",
      "         0.78575           1.80000 \n",
      "         0.62170           1.80000 \n",
      "================================================================================\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "\u001b[33m\u001b[1mTest Broken\u001b[22m\u001b[39m\n",
       "  Expression: #= In[174]:2 =# @test_nowarn sol = solve(prob, SAMIN())"
      ]
     },
     "execution_count": 174,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])\n",
    "@test_broken @test_nowarn sol = solve(prob, SAMIN())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 175,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999999999999\n",
       " 0.9999999999999998"
      ]
     },
     "execution_count": 176,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "using NLopt\n",
    "prob = OptimizationProblem(optprob, x0)\n",
    "sol = solve(prob, Opt(:LN_BOBYQA, 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 177,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.9999999999894374\n",
       " 0.9999999999844783"
      ]
     },
     "execution_count": 178,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol = solve(prob, Opt(:LD_LBFGS, 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 179,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.8\n",
       " 0.6400000000000001"
      ]
     },
     "execution_count": 180,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "prob = OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])\n",
    "sol = solve(prob, Opt(:LD_LBFGS, 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 181,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.8\n",
       " 0.6400000000000001"
      ]
     },
     "execution_count": 182,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sol = solve(prob, Opt(:G_MLSL_LDS, 2), nstart=2, local_method = Opt(:LD_LBFGS, 2), maxiters=10000)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 183,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[32m\u001b[1m   Resolving\u001b[22m\u001b[39m package versions...\n",
      "\u001b[32m\u001b[1m   Installed\u001b[22m\u001b[39m Evolutionary ─ v0.9.0\n",
      "\u001b[32m\u001b[1m    Updating\u001b[22m\u001b[39m `C:\\Users\\lishu\\.julia\\environments\\v1.6\\Project.toml`\n",
      " \u001b[90m [86b6b26d] \u001b[39m\u001b[92m+ Evolutionary v0.9.0\u001b[39m\n",
      "\u001b[32m\u001b[1m    Updating\u001b[22m\u001b[39m `C:\\Users\\lishu\\.julia\\environments\\v1.6\\Manifest.toml`\n",
      " \u001b[90m [86b6b26d] \u001b[39m\u001b[92m+ Evolutionary v0.9.0\u001b[39m\n",
      "\u001b[32m\u001b[1mPrecompiling\u001b[22m\u001b[39m project...\n",
      "\u001b[32m  ✓ \u001b[39mEvolutionary\n",
      "1 dependency successfully precompiled in 8 seconds (172 already precompiled, 4 skipped during auto due to previous errors)\n"
     ]
    }
   ],
   "source": [
    "Pkg.add(\"Evolutionary\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.986821120271575\n",
       " 0.9737506156710922"
      ]
     },
     "execution_count": 186,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "using Evolutionary\n",
    "sol = solve(prob, CMAES(μ =40 , λ = 100),abstol=1e-15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 187,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting optimization with optimizer DiffEvoOpt{FitPopulation{Float64}, RadiusLimitedSelector, BlackBoxOptim.AdaptiveDiffEvoRandBin{3}, RandomBound{ContinuousRectSearchSpace}}\n",
      "0.00 secs, 0 evals, 0 steps\n",
      "\n",
      "Optimization stopped after 10001 steps and 0.03 seconds\n",
      "Termination reason: Max number of steps (10000) reached\n",
      "Steps per second = 384653.51\n",
      "Function evals per second = 376576.59\n",
      "Improvements/step = 0.45310\n",
      "Total function evaluations = 9791\n",
      "\n",
      "\n",
      "Best candidate found: [0.8, 0.64]\n",
      "\n",
      "Fitness: 0.040000000\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "u: 2-element Vector{Float64}:\n",
       " 0.8\n",
       " 0.6399999999935596"
      ]
     },
     "execution_count": 188,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "using BlackBoxOptim\n",
    "prob = GalacticOptim.OptimizationProblem(optprob, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])\n",
    "sol = solve(prob, BBO())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "\u001b[32m\u001b[1mTest Passed\u001b[22m\u001b[39m"
      ]
     },
     "execution_count": 189,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "@test 10*sol.minimum < l1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Defining OptimizationProblems"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "OptimizationProblem(f, x, p = DiffEqBase.NullParameters(),;\n",
    "                    lb = nothing,\n",
    "                    ub = nothing,\n",
    "                    lcons = nothing,\n",
    "                    ucons = nothing,\n",
    "                    kwargs...)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Formally, the OptimizationProblem finds the minimum of f(x,p) with an initial \n",
    "condition x. The parameters p are optional. lb and ub are arrays matching the \n",
    "size of x, which stand for the lower and upper bounds of x, respectively.\n",
    "\n",
    "f is an OptimizationFunction, as defined here. If f is a standard Julia function, it is automatically converted into an OptimizationFunction with NoAD(), i.e., no automatic generation of the derivative functions.\n",
    "\n",
    "Any extra keyword arguments are captured to be sent to the optimizers."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## OptimizationFunction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "OptimizationFunction{iip}(f,adtype=NoAD();\n",
    "                          grad=nothing,\n",
    "                          hess=nothing,\n",
    "                          hv=nothing,\n",
    "                          cons=nothing,\n",
    "                          cons_j=nothing,\n",
    "                          cons_h=nothing)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The keyword arguments are as follows:\n",
    "\n",
    "grad: Gradient\n",
    "\n",
    "hess: Hessian\n",
    "\n",
    "hv: Hessian vector products hv(du,u,p,t,v) = H*v\n",
    "\n",
    "cons: Constraint function\n",
    "\n",
    "cons_j\n",
    "\n",
    "cons_h"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Common Solver Options"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "solve(prob,alg;kwargs...)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The arguments to solve are common across all of the optimizers. These common arguments are:\n",
    "\n",
    "maxiters (the maximum number of iterations)\n",
    "\n",
    "abstol (absolute tolerance)\n",
    "\n",
    "reltol (relative tolerance)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Local Gradient-Based Optimization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "ADAM() is a good default with decent convergence rate. BFGS() can converge faster but is more prone to hitting bad local optima. LBFGS() requires less memory than BFGS and thus can have better scaling."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Flux.Optimise.Descent: Classic gradient descent optimizer with learning rate\n",
    "\n",
    "Flux.Optimise.Momentum: Classic gradient descent optimizer with learning rate and momentum\n",
    "\n",
    "Flux.Optimise.Nesterov: Gradient descent optimizer with learning rate and Nesterov momentum\n",
    "\n",
    "Flux.Optimise.RMSProp: RMSProp optimizer\n",
    "\n",
    "Flux.Optimise.ADAM: ADAM optimizer\n",
    "\n",
    "Flux.Optimise.RADAM: Rectified ADAM optimizer\n",
    "\n",
    "Flux.Optimise.AdaMax: AdaMax optimizer\n",
    "\n",
    "Flux.Optimise.ADAGRad: ADAGrad optimizer\n",
    "\n",
    "Flux.Optimise.ADADelta: ADADelta optimizer\n",
    "\n",
    "Flux.Optimise.AMSGrad: AMSGrad optimizer\n",
    "\n",
    "Flux.Optimise.NADAM: Nesterov variant of the ADAM optimizer\n",
    "\n",
    "Flux.Optimise.ADAMW: ADAMW optimizer"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Local Derivative-Free Optimization"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Derivative-free optimizers are optimizers that can be used even in cases where no derivatives or automatic differentiation is specified. While they tend to be less efficient than derivative-based optimizers, they can be easily applied to cases where defining derivatives is difficult."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Optim.NelderMead: Nelder-Mead optimizer\n",
    "\n",
    "Optim.SimulatedAnnealing: Simulated Annealing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Julia 1.6.0",
   "language": "julia",
   "name": "julia-1.6"
  },
  "language_info": {
   "file_extension": ".jl",
   "mimetype": "application/julia",
   "name": "julia",
   "version": "1.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
