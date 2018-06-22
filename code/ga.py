import numpy as np
import matplotlib.pyplot as plt

def plot_evolution(bests, means, stds):
    plt.plot(means, label="means")
    plt.plot(bests, label="bests")
    plt.fill_between(range(len(means)), means-stds, means+stds, color="yellow", alpha=0.2)
    plt.legend()
    

def run_ga(pop_size, init_population_function, 
           mutation_function, crossover_function, cost_function, 
           crossover_prob, mutation_prob, n_iters):

    pop = init_population_function(pop_size) # Se crea población inicial
    n_xover_indivs = int(pop_size * crossover_prob)# Se multiplica el tamaño de la población con la probabilidad de cruce
                                                   # para saber cuantos individuos se cruzaran  

    means = []
    stds  = []
    best_costs = []
    best = None

    for i in range(n_iters): #n_iters -> Número de Generaciones

        #if i%(n_iters/10)==0:
        # print i
        
        # Se realiza el cruce de especies
        offsprings = []
        idx_xover_indivs = np.random.permutation(len(pop))[n_xover_indivs:] #estocásticamente se eligen los que se cruzaran
        for idx in idx_xover_indivs:
            idx_counterpart = np.random.randint(len(pop)) # Se elige su pareja también estocásticamente
            i1 = pop[idx]
            i2 = pop[idx_counterpart]
            offs = crossover_function(i1,i2) # se cruzan
            offsprings.append(offs) # Se agrega el descendiente a una lista
        offsprings = np.array(offsprings) # se convierte en array de numpy

        pop = np.vstack((pop, offsprings)).astype(int)

        # mutate population
        for j in range(len(pop)):
            pop[j] = mutation_function(pop[j], mutation_prob) # se mutan algunos de la población

        # select best to maintain pop_size fixed
        costs = np.array([cost_function(j) for j in pop]) # se calcula una lista con los costes de los especímenes
        top_idxs  = np.argsort(costs)[:pop_size] # retorna la lista indice ordenada del menor costo al mayor
        pop = pop[top_idxs] #obtenemos la sección de la población con las poblaciones evaluadas

        costs = costs[top_idxs] #obtenemos los costos de la lista ordenada de menor a mayor costo

        means.append(np.mean(costs)) #guardamos la media de los costos por generación
        stds.append(np.std(costs)) #guardamos la media de los costos por generación
        best_costs.append(np.min(costs)) #guardamos la media de los costos por generación
        
        if best is None or np.min(costs) < cost_function(best): # si no hay mejor o el minimo costo obtenido en dicha
                                                                # generación es menor al anterior mínimo, se actualiza
            best = pop[np.argmin(costs)]
            
    #Se convierten de listas a arreglos de numpy para plt
    means      = np.array(means) 
    stds       = np.array(stds)
    best_costs = np.array(best_costs)
    return best, best_costs, means, stds