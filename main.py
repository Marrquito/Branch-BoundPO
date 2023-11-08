from mip import *



def main(method: int): 
    coefficienctsObjectEquation         = []
    coefficienctsRestrictionVar         = []
    coefficientsRightSideRestriction    = []
    qtdVars        : int                = 0
    qtdRestricts   : int                = 0
    
    try:
    
        match method:
            case 1:
                qtdVars, qtdRestricts, coefficienctsObjectEquation = read_values_by_console(coefficienctsRestrictionVar, coefficientsRightSideRestriction)
                
            case 2:
                qtdVars, qtdRestricts, coefficienctsObjectEquation = read_values_by_file(coefficienctsRestrictionVar, coefficientsRightSideRestriction)
    except Exception as e:
        print(f"Error reading {e}")
        return
    
    # print("\nDados de entrada:\n")
    print(f"Número de variáveis:                {qtdVars} e restrições: {qtdRestricts}")
    print(f"Coeficientes da função objetivo:    {coefficienctsObjectEquation}")
    print(f"Coeficientes das restrições:        {coefficienctsRestrictionVar}")
    print(f"Lados direitos das restrições:      {coefficientsRightSideRestriction}")
    
    print("\n")
    print("Iniciando procedimento de branch and bound...\n")
    best_solution, best_vars = branch_and_bound(qtdVars, coefficienctsObjectEquation, coefficienctsRestrictionVar, coefficientsRightSideRestriction)
    
    print("Melhor solução encontrada: ")
    print("Valor da Função Objetivo:", best_solution)
    print("Valores das variáveis:", best_vars)

def create_model(qtdVars: int, coefficienctsObjectEquation: list[int], coefficienctsRestrictionVar: list[int], coefficientsRightSideRestriction: list[int]):
    model = Model(sense=MAXIMIZE, solver_name=CBC)
    
    # Variáveis binárias
    x = [model.add_var(var_type=BINARY) for _ in range(qtdVars)]
    
    # Função objetivo
    model.objective = xsum(coefficienctsObjectEquation[i] * x[i] for i in range(qtdVars))

    # Restrições
    for j in range(len(coefficientsRightSideRestriction)):
        model += xsum(coefficienctsRestrictionVar[j][i] * x[i] for i in range(qtdVars)) <= coefficientsRightSideRestriction[j]

    return model

# Algoritmo Branch and Bound
def branch_and_bound(qtdVars: int, coefficienctsObjectEquation: list[int], coefficienctsRestrictionVar: list[int], coefficientsRightSideRestriction: list[int]):
    best_solution   = 0  # Melhor solução encontrada até agora
    best_vars       = None  # Variáveis correspondentes à melhor solução
    queue           = [create_model(qtdVars, coefficienctsObjectEquation, coefficienctsRestrictionVar, coefficientsRightSideRestriction)]  # Fila de modelos a serem resolvidos

    while queue:
        model = queue.pop(0)
        model.optimize()

        if model.num_solutions:
            solution    = model.vars
            objective   = model.objective_value

            if objective > best_solution:
                best_solution   = objective
                best_vars       = [int(round(var.x)) for var in solution]

            fractional_var = None
            for var in solution:
                if var.x not in (0, 1):
                    fractional_var = var
                    break

            if fractional_var:
                # Ramificar em torno da variável fracionária
                new_model1  = create_model()
                new_model1 += fractional_var <= 0
                queue.append(new_model1)

                new_model2  = create_model()
                new_model2 += fractional_var >= 1
                queue.append(new_model2)

    return best_solution, best_vars

def read_values_by_console(coefficienctsRestrictionVar, coefficientsRightSideRestriction):
    # listas para armazenar os coeficientes da função objetivo, coeficientes das restrições (coefficienctsRestrictionVar) e lados direitos das restrições (coefficientsRightSideRestriction)
    
    qtdVars, qtdRestricts       = map(int, input("Número de variáveis e número de restrições (separados por espaço): ").split())   
    coefficienctsObjectEquation = list(map(float, input("Insira os coeficientes da função objetivo (separados por espaço): ").split()))

    print(f"Insira os coeficientes das {qtdRestricts} restrições e os lados direitos (separados por espaço):")
    for i in range(qtdRestricts):
        coefficients_and_b = list(map(float, input(f"{i+1}: ").split()))
        coefficienctsRestrictionVar.append(coefficients_and_b[:-1])
        coefficientsRightSideRestriction.append(coefficients_and_b[-1])

    return qtdVars, qtdRestricts, coefficienctsObjectEquation

def read_values_by_file(coefficienctsRestrictionVar, coefficientsRightSideRestriction):
    file_name = input("Digite o nome do arquivo a ser lido: ")
    
    with open(file_name, "r") as file:
        qtdVars, qtdRestricts       = map(int, file.readline().split())
        coefficienctsObjectEquation = list(map(float, file.readline().split()))
        
        for i in range(qtdRestricts):
            coefficients_and_b = list(map(float, file.readline().split()))
            coefficienctsRestrictionVar.append(coefficients_and_b[:-1])
            coefficientsRightSideRestriction.append(coefficients_and_b[-1])
    
    return qtdVars, qtdRestricts, coefficienctsObjectEquation

if __name__ == "__main__":
    print("Bem-vindos ao seu programa de Branch & Bound")
    
    method = int(input("Voce gostaria de inserir os dados manualmente (1) ou via arquivo (2)? "))
    
    print("")
    
    main(method)