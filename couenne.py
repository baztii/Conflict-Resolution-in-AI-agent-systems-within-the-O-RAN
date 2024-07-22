from pyomo.environ import SolverFactory

solver = SolverFactory('couenne')
if not solver.available():
    raise RuntimeError("Couenne solver is not available. Make sure it is installed and accessible.")

exit()


from pyomo.environ import *

# Crear un simple modelo de prueba
model = ConcreteModel()
model.x = Var(bounds=(0, 10), initialize=5)
model.obj = Objective(expr=model.x**2)
model.con = Constraint(expr=model.x >= 1)

# Intentar resolver el modelo usando Couenne
solver = SolverFactory('couenne')

# Verificar si el solver está disponible
if not solver.available():
    raise RuntimeError("Couenne solver is not available. Make sure it is installed and accessible.")

results = solver.solve(model, tee=True)
model.display()