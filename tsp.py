import sys
from typing import List, Tuple
from random import seed, randint
from itertools import product
from math import sqrt
import mip
from mip import Model, xsum, BINARY, CONTINUOUS, minimize, CBC, OptimizationStatus
from time import time
import numpy as np
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import pandas as pd
seed(0)

TIME_LIMIT = 60.0  # segundos (1 minuto)

def main():
    n = 31
    solver_names = ["CBC", "gurobi", "HIGHS"]
    formulation_names = ["singlecommodity", "multicommodity", "twocommodity",
                         "arcindexed","arcindexed2", "disjunctive", "twoindexedcutloop"]
    verbose = False
    solutions = {(s,f): [] for s in solver_names for f in formulation_names}
    solutions["n"] = []
    for h in range(10, n, 5):
        dt = CData(h)
        solutions["n"].append(h)
        for solver_name in solver_names:
            print(f"\n\n {solver_name}")
            for formulation_name in formulation_names:
                match formulation_name:
                    case "twoindexedcutloop":
                        cutloop = CCutLoopModelTSP(dt, formulation_name, solver_name, verbose)
                        rt = cutloop.run(time_limit=TIME_LIMIT)
                    case "singlecommodity":
                        single = CSingleCommodityTSP(dt, formulation_name, solver_name, verbose)
                        rt = single.run(time_limit=TIME_LIMIT)
                    case "arcindexed":
                        arcindexed = CArcIndexedTSP(dt, formulation_name, solver_name, verbose)
                        rt = arcindexed.run(time_limit=TIME_LIMIT)
                    case "arcindexed2":
                        arcindexed2 = CArcIndexedTSP2(dt, formulation_name, solver_name, verbose)
                        rt = arcindexed2.run(time_limit=TIME_LIMIT)
                    case "multicommodity":
                        multicommodity = MulticommodityTSPModel(dt, formulation_name, solver_name, verbose)
                        rt = multicommodity.run(time_limit=TIME_LIMIT)
                    case "twocommodity":
                        twocommodity = TwoCommodityTSPModel(dt, formulation_name, solver_name, verbose)
                        rt = twocommodity.run(time_limit=TIME_LIMIT)
                    case "disjunctive":
                        disjunctive = DisjunctiveTSPModel(dt, formulation_name, solver_name, verbose)
                        rt = disjunctive.run(time_limit=TIME_LIMIT)
                solutions[(solver_name, formulation_name)].append(rt)
    df = pd.DataFrame(solutions)
    df.set_index(df.columns[-1], inplace=True)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["solver","formulation"])
    print("\n")
    print(df)
    ax = df.plot.bar(rot=0, log=True, figsize=(8,6))
    plt.savefig("comparison.pdf")
    print("\n")
    plt.show()

class CData:
    def __init__(self, n):
        self.n = n
        V = set(range(n))
        p = [(randint(1, 100), randint(1, 100)) for _ in V]  # coordenadas
        A = [(i, j) for (i, j) in product(V, V) if i != j]
        c = [[round(sqrt((p[i][0]-p[j][0])**2 + (p[i][1]-p[j][1])**2)) for j in V] for i in V]
        self.V, self.A = V, A
        self.p = p
        self.c = c

class CCutLoopModelTSP:
    def __init__(self, dt: CData, formulation_name: str, solver_name: str, verbose: bool):
        self.model_name = formulation_name
        self.dt = dt
        V,A,c,n = dt.V,dt.A,dt.c,dt.n

        model = Model(solver_name=solver_name)
        model.verbose = verbose

        x = {(i,j): model.add_var(var_type=BINARY) for j in V for i in V if i != j}
        model.objective = minimize(xsum(c[i][j]*x[i,j] for (i, j) in A))

        for i in V:
            model += xsum(x[i,j] for (ii,j) in A if ii == i) == 1
        for j in V:
            model += xsum(x[i,j] for (i,jj) in A if jj == j) == 1
        for (i, j) in A:
            model += x[i,j] + x[j,i] <= 1

        self.model, self.x = model, x
        self.graph = np.zeros((n,n))

    def run(self, time_limit: float = TIME_LIMIT):
        model, x, g = self.model, self.x, self.graph
        dt = self.dt
        V,A,c,n = dt.V,dt.A,dt.c,dt.n

        start = time()
        deadline = start + time_limit
        it = 0
        print(f"\n\n {self.model_name}")
        stop = False

        while not stop:
            it += 1
            remaining = max(0.0, deadline - time())
            if remaining <= 0.0:
                print(f" tempo limite atingido ({time_limit:.0f}s) — resultado não alcançado")
                return time_limit
            model.max_seconds = remaining

            status = model.optimize()
            rt = time() - start

            if status == OptimizationStatus.OPTIMAL:
                lb = model.objective_value
                for (i,j) in A:
                    g[i][j] = x[i,j].x
                cc = connected_components(g, directed=True)
                if cc[0] > 1:
                    for comp in range(cc[0]):
                        S = [j for j in V if cc[1][j] == comp]
                        AS = [(i,j) for (i,j) in product(S,S) if i != j]
                        model += xsum(x[i,j] for (i,j) in AS) <= len(S) - 1
                else:
                    stop = True
                print(f'{it:3d} {lb:12.2f} {rt:12.2f} s')
            elif status == OptimizationStatus.TIME_LIMIT:
                print(f" tempo limite atingido ({time_limit:.0f}s) — resultado não alcançado")
                return time_limit
            else:
                print(' status       : %s ' % model.status)
                return time_limit  # considera não alcançado

        rt = min(time() - start, time_limit)
        print(" status       : %s " % model.status)
        print(" tour         : {:12.2f}.".format(model.objective_value))
        print(" running time : {:.2f} s".format(rt))
        return rt

class CArcIndexedTSP2:
    def __init__(self, dt: CData, formulation_name: str, solver_name: str, verbose: bool):
        self.model_name = formulation_name
        self.dt = dt
        V,A,c,n = dt.V,dt.A,dt.c,dt.n
        K = range(n)
        model = Model(solver_name=solver_name)
        model.verbose = verbose

        x = {(i,j): model.add_var(var_type=BINARY, name=f'x_{i}_{j}') for (i,j) in A}
        z = {(i,j,k): model.add_var(var_type=BINARY, name=f'z_{i}_{j}_{k}') for (i,j) in A for k in K}

        model.objective = minimize(xsum(c[i][j]*z[i,j,k] for (i, j) in A for k in K))
        for i in V:
            model += xsum(z[i,j,k] for (ii,j) in A if ii == i for k in K) == 1
        for j in V:
            model += xsum(z[i,j,k] for (i,jj) in A if jj == j for k in K) == 1
        model += xsum(z[i,j,0] for (i,j) in A if i == 0) == 1
        for j in V:
            if j != 0:
                for k in K:
                    if k < n-1:
                        model += xsum(z[i,jj,k] for (i,jj) in A if jj == j) - \
                                 xsum(z[jj,i,k+1] for (jj,i) in A if jj == j) == 0

        self.model, self.z = model, z

    def run(self, time_limit: float = TIME_LIMIT):
        print(f"\n\n {self.model_name}")
        model = self.model
        model.max_seconds = time_limit
        start = time()
        status = model.optimize()
        end = time()
        if status == OptimizationStatus.OPTIMAL:
            print(' status       : %s ' % model.status)
            print(" tour         : {:12.2f}.".format(model.objective_value))
            print(" running time : {:.2f} s".format(end-start))
            return end-start
        else:
            print(f" tempo limite/sem ótimo — resultado não alcançado (status={model.status})")
            return time_limit

class CArcIndexedTSP:
    def __init__(self, dt: CData, formulation_name: str, solver_name: str, verbose: bool):
        self.model_name = formulation_name
        self.dt = dt
        V,A,c,n = dt.V,dt.A,dt.c,dt.n
        K = range(n)
        model = Model(solver_name=solver_name)
        model.verbose = verbose

        x = {(i,j): model.add_var(var_type=BINARY, name=f'x_{i}_{j}') for (i,j) in A}
        z = {(i,j,k): model.add_var(var_type=BINARY, name=f'z_{i}_{j}_{k}') for (i,j) in A for k in K}

        model.objective = minimize(xsum(c[i][j]*x[i,j] for (i, j) in A))
        for i in V:
            model += xsum(x[i,j] for (ii,j) in A if ii == i) == 1
        for j in V:
            model += xsum(x[i,j] for (i,jj) in A if jj == j) == 1
        model += xsum(z[i,j,0] for (i,j) in A if i == 0) == 1
        for (i,j) in A:
            model += xsum(z[i,j,k] for k in K) == x[i,j]
        for j in V:
            if j != 0:
                for k in K:
                    if k < n-1:
                        model += xsum(z[i,jj,k] for (i,jj) in A if jj == j) - \
                                 xsum(z[jj,i,k+1] for (jj,i) in A if jj == j) == 0

        self.model, self.x, self.z = model, x, z

    def run(self, time_limit: float = TIME_LIMIT):
        print(f"\n\n {self.model_name}")
        model = self.model
        model.max_seconds = time_limit
        start = time()
        status = model.optimize()
        end = time()
        if status == OptimizationStatus.OPTIMAL:
            print(' status       : %s ' % model.status)
            print(" tour         : {:12.2f}.".format(model.objective_value))
            print(" running time : {:.2f} s".format(end-start))
            return end-start
        else:
            print(f" tempo limite/sem ótimo — resultado não alcançado (status={model.status})")
            return time_limit

class CSingleCommodityTSP:
    def __init__(self, dt: CData, formulation_name: str, solver_name: str, verbose: bool):
        self.model_name = formulation_name
        self.dt = dt
        V,A,c,n = dt.V,dt.A,dt.c,dt.n

        model = Model(solver_name=solver_name)
        model.verbose = verbose

        x = {(i,j): model.add_var(var_type=BINARY) for (i,j) in A}
        f = {(i,j): model.add_var(var_type=CONTINUOUS) for (i,j) in A}

        model.objective = minimize(xsum(c[i][j]*x[i,j] for (i, j) in A))
        for i in V:
            model += xsum(x[i,j] for (ii,j) in A if ii == i) == 1
        for j in V:
            model += xsum(x[i,j] for (i,jj) in A if jj == j) == 1
        model += xsum(f[i,j] for (i,j) in A if i == 0) == n - 1
        for j in V:
            if j != 0:
                model += xsum(f[i,jj] for (i,jj) in A if jj == j) - \
                         xsum(f[jj,i] for (jj,i) in A if jj == j) == 1
        for (i, j) in A:
            model += f[i,j] <= (n-1) * x[j,i]

        self.model, self.x, self.f = model, x, f

    def run(self, time_limit: float = TIME_LIMIT):
        print(f"\n\n {self.model_name}")
        model = self.model
        model.max_seconds = time_limit
        start = time()
        status = model.optimize()
        end = time()
        if status == OptimizationStatus.OPTIMAL:
            print(' status       : %s ' % model.status)
            print(" tour         : {:12.2f}.".format(model.objective_value))
            print(" running time : {:.2f} s".format(end-start))
            return end-start
        else:
            print(f" tempo limite/sem ótimo — resultado não alcançado (status={model.status})")
            return time_limit

class MulticommodityTSPModel:
    def __init__(self, dt: CData, formulation_name: str, solver_name: str, verbose: bool):
        self.model_name = formulation_name
        self.dt = dt
        V,A,c,n = dt.V,dt.A,dt.c,dt.n
        K = [k for k in V if k != 0]

        model = Model(solver_name=solver_name)
        model.verbose = verbose

        x = {(i,j): model.add_var(var_type=BINARY) for j in V for i in V if i != j}
        f = {(k,i,j): model.add_var(var_type=CONTINUOUS, lb=0.0, name=f"f_{k}_{i}_{j}")
             for k in K for (i, j) in A}

        model.objective = minimize(xsum(c[i][j]*x[i,j] for (i, j) in A))
        for i in V:
            model += xsum(x[i, j] for (ii, j) in A if ii == i) == 1
        for j in V:
            model += xsum(x[i, j] for (i, jj) in A if jj == j) == 1

        for k in K:
            model += xsum(f[k, i, j] for (i, j) in A if i == 0) == 1
            for j in V:
                if j == 0:
                    continue
                incoming = xsum(f[k, i, jj] for (i, jj) in A if jj == j)
                outgoing = xsum(f[k, jj, i] for (jj, i) in A if jj == j)
                rhs = 1 if j == k else 0
                model += incoming - outgoing == rhs
            for (i, j) in A:
                model += f[k, i, j] <= x[i, j]

        self.model, self.x, self.f = model, x, f
        self.graph = np.zeros((n, n))

    def run(self, time_limit: float = TIME_LIMIT):
        print(f"\n\n {self.model_name}")
        model = self.model
        model.max_seconds = time_limit
        start = time()
        status = model.optimize()
        end = time()
        if status == OptimizationStatus.OPTIMAL:
            print(' status       : %s ' % model.status)
            print(" tour         : {:12.2f}.".format(model.objective_value))
            print(" running time : {:.2f} s".format(end-start))
            return end-start
        else:
            print(f" tempo limite/sem ótimo — resultado não alcançado (status={model.status})")
            return time_limit

class TwoCommodityTSPModel:
    def __init__(self, dt, formulation_name: str, solver_name: str, verbose: bool):
        self.model_name = formulation_name
        self.dt = dt
        V, A, c, n = dt.V, dt.A, dt.c, dt.n

        model = Model(solver_name=solver_name)
        model.verbose = verbose

        x = {(i, j): model.add_var(var_type=BINARY, name=f"x_{i}_{j}") for (i, j) in A}
        g = {(i, j): model.add_var(var_type=CONTINUOUS, name=f"g_{i}_{j}") for (i, j) in A}
        h = {(i, j): model.add_var(var_type=CONTINUOUS, name=f"h_{i}_{j}") for (i, j) in A}

        model.objective = minimize(xsum(c[i][j] * x[i, j] for (i, j) in A))
        for i in V:
            model += xsum(x[i, j] for (ii, j) in A if ii == i) == 1
        for j in V:
            model += xsum(x[i, j] for (i, jj) in A if jj == j) == 1

        for (i, j) in A:
            if (j, i) in A and i < j:
                model += x[i, j] + x[j, i] <= 1

        model += xsum(g[i, j] for (i, j) in A if i == 0) == n - 1
        for v in V:
            if v == 0: continue
            incoming = xsum(g[i, j] for (i, j) in A if j == v)
            outgoing = xsum(g[i, j] for (i, j) in A if i == v)
            model += incoming - outgoing == 1

        model += xsum(h[i, j] for (i, j) in A if j == 0) == n - 1
        for v in V:
            if v == 0: continue
            incoming = xsum(h[i, j] for (i, j) in A if j == v)
            outgoing = xsum(h[i, j] for (i, j) in A if i == v)
            model += incoming - outgoing == -1

        undirected = {(min(i, j), max(i, j)) for (i, j) in A}
        for (i, j) in undirected:
            model += g[i, j] + g[j, i] + h[i, j] + h[j, i] <= (n - 1) * (x[i, j] + x[j, i])

        for (i, j) in A:
            model += g[i, j] <= (n - 1) * x[i, j]
            model += h[i, j] <= (n - 1) * x[i, j]

        self.model, self.x, self.g, self.h = model, x, g, h
        self.graph = np.zeros((n, n))

    def run(self, time_limit: float = TIME_LIMIT):
        print(f"\n\n {self.model_name}")
        model = self.model
        model.max_seconds = time_limit
        start = time()
        status = model.optimize()
        end = time()
        if status == OptimizationStatus.OPTIMAL:
            print(' status       : %s ' % model.status)
            print(" tour         : {:12.2f}.".format(model.objective_value))
            print(" running time : {:.2f} s".format(end-start))
            return end-start
        else:
            print(f" tempo limite/sem ótimo — resultado não alcançado (status={model.status})")
            return time_limit

class DisjunctiveTSPModel:
    def __init__(self, dt, formulation_name: str, solver_name: str, verbose: bool):
        self.model_name = formulation_name
        self.dt = dt
        V, A, c, n = dt.V, dt.A, dt.c, dt.n

        model = Model(solver_name=solver_name)
        model.verbose = verbose

        x = {(i, j): model.add_var(var_type=BINARY, name=f"x_{i}_{j}") for (i, j) in A}
        t = {j: model.add_var(var_type=CONTINUOUS, lb=1.0, ub=n-1, name=f"t_{j}")
             for j in V if j != 0}

        model.objective = minimize(xsum(c[i][j] * x[i, j] for (i, j) in A))
        for i in V:
            model += xsum(x[i, j] for (ii, j) in A if ii == i) == 1
        for j in V:
            model += xsum(x[i, j] for (i, jj) in A if jj == j) == 1

        for (i, j) in A:
            if (j, i) in A and i < j:
                model += x[i, j] + x[j, i] <= 1

        M = n - 1
        for (i, j) in A:
            if i != 0 and j != 0 and i != j:
                model += t[j] >= t[i] + 1 - M * (1 - x[i, j])

        self.model, self.x, self.t = model, x, t
        self.graph = np.zeros((n, n))

    def run(self, time_limit: float = TIME_LIMIT):
        print(f"\n\n {self.model_name}")
        model = self.model
        model.max_seconds = time_limit
        start = time()
        status = model.optimize()
        end = time()
        if status == OptimizationStatus.OPTIMAL:
            print(' status       : %s ' % model.status)
            print(" tour         : {:12.2f}.".format(model.objective_value))
            print(" running time : {:.2f} s".format(end-start))
            return end-start
        else:
            print(f" tempo limite/sem ótimo — resultado não alcançado (status={model.status})")
            return time_limit

if __name__ == '__main__':
    main()