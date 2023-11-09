from   mip      import Model, xsum, CBC, MAXIMIZE, CONTINUOUS
import numpy    as np
import os

class BranchAndBound():
    def __init__(self):
        self.CLOSEST_VALUE_XJ   = 0.5
        self.primal             = 0
        
        # variaveis referentes a melhor solução  
        self.best_solution   = 0   
        self.best_vars       = []  
        
    def main(self, method: int): 
        coefficienctsObjectEquation         = []
        coefficienctsRestrictionVar         = []
        coefficientsRightSideRestriction    = []
        qtdVars        : int                = 0
        qtdRestricts   : int                = 0
        
        try:
            match method:
                case 1:
                    qtdVars, qtdRestricts, coefficienctsObjectEquation = self.read_values_by_console(coefficienctsRestrictionVar, coefficientsRightSideRestriction)
                    
                case 2:
                    qtdVars, qtdRestricts, coefficienctsObjectEquation = self.read_values_by_file(coefficienctsRestrictionVar, coefficientsRightSideRestriction)
        except FileNotFoundError as e:
            print(f"Error reading file: {e}")
            
            return
        except Exception as e:
            print(f"Error: {e}")
            
            return
        
        print("\nDados de entrada:\n")
        print(f"Número de variáveis:                {qtdVars} e restrições: {qtdRestricts}")
        print(f"Coeficientes da função objetivo:    {coefficienctsObjectEquation}")
        print(f"Coeficientes das restrições:        {coefficienctsRestrictionVar}")
        print(f"Lados direitos das restrições:      {coefficientsRightSideRestriction}")
        print("\n")
        
        print("Iniciando procedimento de branch and bound...\n")
        self.branch_and_bound(qtdVars, coefficienctsObjectEquation, coefficienctsRestrictionVar, coefficientsRightSideRestriction)
        
        print("Melhor solução encontrada: ")
        print(f"Valor da Função Objetivo:   {self.best_solution}")
        print(f"Valores das variáveis:      {self.best_vars}")

    def create_model(self, qtdVars: int, coefficienctsObjectEquation: list[int], coefficienctsRestrictionVar: list[int], coefficientsRightSideRestriction: list[int]):
        model = Model(sense=MAXIMIZE, solver_name=CBC)

        # variavel continua
        x               = [model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name="x_" + str(i)) for i in range(qtdVars)]
        
        # função objetivo
        model.objective = xsum(coefficienctsObjectEquation[i] * x[i] for i in range(qtdVars))

        # restrições
        for j in range(len(coefficientsRightSideRestriction)):
            model += xsum(coefficienctsRestrictionVar[j][i] * x[i] for i in range(qtdVars)) <= coefficientsRightSideRestriction[j]

        return model

    def branch_and_bound(self, qtdVars: int, coefficienctsObjectEquation: list[int], coefficienctsRestrictionVar: list[int], coefficientsRightSideRestriction: list[int]):        
        # fila de modelos a serem resolvidos
        queue = [self.create_model(qtdVars, coefficienctsObjectEquation, coefficienctsRestrictionVar, coefficientsRightSideRestriction)]  

        while queue:
            mode, solution_vars, objective = self.bound(queue[0])

            if mode in ["INVIABILIDADE", "LIMITANTE"]:
                queue.pop(0)
            elif mode == "INTEGRALIDADE":
                if objective is not None and objective > self.primal:
                    self.primal         = objective
                    self.best_solution  = objective
                    self.best_vars      = [int(round(var.x)) for var in solution_vars]
                
                queue.pop(0)
            elif mode == "FRACIONARIO":
                var_branch      = solution_vars[self.closest_value([i.x for i in solution_vars], self.CLOSEST_VALUE_XJ)]
                original_model  = queue.pop(0)
                
                new_model1 = original_model.copy()   # restrição var == 0
                new_model1 += var_branch == 0
                
                new_model2 = original_model.copy()   # restrição var == 1
                new_model2 += var_branch == 1
                
                queue.append(new_model1)
                queue.append(new_model2)
    
    # verifica se o modelo resolvido é inviável, limitante, fracionário ou integral
    def bound(self, model): 
        model.optimize()
        
        fractional_var  = False
        solution_vars   = model.vars
        objective       = model.objective_value
                
        if not objective: 
            return 'INVIABILIDADE', [], None

        if objective <= self.primal: 
            return 'LIMITANTE', [], None
        
        # verificando se alguma variável tem valor fracionário
        for var in solution_vars: 
            if var.x != int(var.x):
                fractional_var = True
                break
        
        if not fractional_var: # retorna integralidade se não for fracionário 
            print("Encontrada integralidade")
            return 'INTEGRALIDADE', solution_vars, objective

        return 'FRACIONARIO', solution_vars, objective # caso não seja nenhuma das outras opções, retorna fracionário para ser ramificado
   
    # retorna o valor mais proximo da list a partir do valor informado
    def closest_value(self, array, value): 
        array       = np.asarray(array)
        
        value_found = np.absolute(array - value)
        value_found = value_found.argmin()
        
        return value_found 
    
    def read_values_by_console(self, coefficienctsRestrictionVar, coefficientsRightSideRestriction):
        qtdVars, qtdRestricts       = map(int, input("Número de variáveis e número de restrições (separados por espaço): ").split())   
        coefficienctsObjectEquation = list(map(int, input("Insira os coeficientes da função objetivo (separados por espaço): ").split()))

        print(f"Insira os coeficientes das {qtdRestricts} restrições e os lados direitos (separados por espaço):")
        for i in range(qtdRestricts):
            coefficients_and_b = list(map(int, input(f"{i+1}: ").split()))
            coefficienctsRestrictionVar.append(coefficients_and_b[:-1])
            coefficientsRightSideRestriction.append(coefficients_and_b[-1])

        return qtdVars, qtdRestricts, coefficienctsObjectEquation

    def read_values_by_file(self, coefficienctsRestrictionVar, coefficientsRightSideRestriction):
        file_name = input("Digite o nome do arquivo a ser lido: ")
        
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        
        with open(file_name, "r") as file:
            qtdVars, qtdRestricts       = map(int, file.readline().split())
            coefficienctsObjectEquation = list(map(int, file.readline().split()))
            
            for _ in range(qtdRestricts):
                coefficients_and_b = list(map(int, file.readline().split()))
                coefficienctsRestrictionVar.append(coefficients_and_b[:-1])
                coefficientsRightSideRestriction.append(coefficients_and_b[-1])
        
        return qtdVars, qtdRestricts, coefficienctsObjectEquation

if __name__ == "__main__":
    print("Bem-vindos ao seu programa de Branch & Bound")
    
    method = int(input("Voce gostaria de inserir os dados manualmente (1) ou via arquivo (2)? "))
    
    print("")
    
    bb = BranchAndBound()
    bb.main(method)