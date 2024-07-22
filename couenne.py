from pyomo.opt import SolverFactory

# Verificar disponibilidad del solver SCIP
solver = SolverFactory('scip')
if solver.available():
    print("SCIP solver is available.")
else:
    print("SCIP solver is not available.")