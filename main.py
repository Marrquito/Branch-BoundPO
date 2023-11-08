from mip import Model, xsum, BINARY, OptimizationStatus

# Solicita ao usuário o número de variáveis e restrições
n, m = map(int, input("Número de variáveis e número de restrições (separados por espaço): ").split())

# Inicializa listas para armazenar os coeficientes da função objetivo, coeficientes das restrições (A) e lados direitos das restrições (b)
c = []
A = []
b = []

# Solicita ao usuário os coeficientes da função objetivo
print("Insira os coeficientes da função objetivo (separados por espaço):")
c = list(map(float, input().split()))

# Solicita ao usuário os coeficientes das restrições e lados direitos
print(f"Insira os coeficientes das {m} restrições e os lados direitos (separados por espaço):")
for i in range(m):
    coefficients_and_b = list(map(float, input().split()))
    A.append(coefficients_and_b[:-1])
    b.append(coefficients_and_b[-1])

# Impressão dos dados de entrada
print("Dados de entrada:")
print("Número de variáveis e restrições:", n, m)
print("Coeficientes da função objetivo:", c)
print("Coeficientes das restrições (A):", A)
print("Lados direitos das restrições (b):", b)

# Função para criar um modelo MIP
def create_model():
    model = Model()

    # Variáveis binárias
    x = [model.add_var(var_type=BINARY) for _ in range(n)]

    # Função objetivo
    model.objective = xsum(c[i] * x[i] for i in range(n))

    # Restrições
    for j in range(len(b)):
        model += xsum(A[j][i] * x[i] for i in range(n)) <= b[j]

    return model

# Algoritmo Branch and Bound
def branch_and_bound():
    best_solution = 0  # Melhor solução encontrada até agora
    best_vars = None  # Variáveis correspondentes à melhor solução
    queue = [create_model()]  # Fila de modelos a serem resolvidos

    while queue:
        model = queue.pop(0)
        model.optimize()

        if model.num_solutions:
            solution = model.vars
            objective = model.objective_value

            if objective > best_solution:
                best_solution = objective
                best_vars = [int(round(var.x)) for var in solution]

            fractional_var = None
            for var in solution:
                if var.x not in (0, 1):
                    fractional_var = var
                    break

            if fractional_var:
                # Ramificar em torno da variável fracionária
                new_model1 = create_model()
                new_model1 += fractional_var <= 0
                queue.append(new_model1)

                new_model2 = create_model()
                new_model2 += fractional_var >= 1
                queue.append(new_model2)

    return best_solution, best_vars

best_solution, best_vars = branch_and_bound()

print("Melhor solução encontrada:")
print("Valor da Função Objetivo:", best_solution)
print("Valores das variáveis:", best_vars)
