
import numpy as np
import matplotlib.pyplot as plt

def plot_evolution(bests, means, stds):
    plt.plot(means, label="means")
    plt.plot(bests, label="bests")
    plt.fill_between(range(len(means)), means-stds, means+stds, color="yellow", alpha=0.2)
    plt.legend()

def run_sa(n_individuals, n_cooling_steps, init_population_function, cost_function, generate_neighbor_function):
          #         run_ga                            run_sa
        
          #        pop_size           =           n_individuals 
    
          # mutation_function                    # generate_neighbor_function
          # crossover_function            
          # crossover_prob
          # mutation_prob 
          # n_iters                   =            n_cooling_steps

    pop = init_population_function(n_individuals) # Crea la poblacion inicial (n_individuals vectores de soluciones)
    # GA -> se determina el numero de individos a cruzar (n_xover_indivs)
    
    mean_costs = []
    std_costs  = []
    best_costs = []
    best_sols  = []

    min_cost = np.inf
    min_sol  = None

    # GA:  se itera según el Número de Generaciones (n_iters)
    
    for T in np.linspace(1,0,n_cooling_steps): # array numpy que empieza en uno y para en cero de len(n_cooling_steps)
        costs = []                             # para cada paso de enfriamiento se declara una lista de costos
        for i in range(len(pop)):              # recorremos las ciudades
            sol = pop[i]                       
            cost_sol = cost_function(sol)      # hallamos costo a cada individuo de la poblacion ->(costo a solucion)

            # generate a neighbour
            nbr = generate_neighbor_function(sol) # un especimen solucion parecido 1 bit mutado
            cost_nbr = cost_function(nbr)         # se le halla el coste a la mutacion

            # if the neighbour is better
            if cost_nbr<cost_sol or np.random.random()<T: # si el costo es menor lo reemplazamos o si estocásticamente
                sol = nbr                                 # es menor que el paso de enfriamiento, el cual se hace cada vez                       
                cost_sol = cost_nbr                       # mas pequeño, por lo tanto es mas dificil que sea una mutacion
                                                          # con coste mayor reemplazada por la original

            pop[i] = sol                                  # Agregamos a la poblacion el individuo solucion
            costs.append(cost_sol)                        # A los costos le agregamos el costo del especímen

            if cost_sol < min_cost:                       # si el costo de la solucion es menor que el mínimo costo hallado
                min_sol  = np.copy(pop[i])                # la solucion mínima sera el especimen
                min_cost = cost_function(pop[i])          # su costo será el mínimo también

        best_costs.append(np.min(costs))                  # agregamos por cada n_cooling_steps el mínimo a la best cost list
        mean_costs.append(np.mean(costs))                 # agregamos por cada n_cooling_steps la media a la mean cost list
        std_costs.append(np.std(costs))                   # agregamos por cada n_cooling_steps el std a la std cost list
 
                                                          # Los convertimos en np.array y los retornamos
    mean_costs = np.array(mean_costs)
    std_costs  = np.array(std_costs)
    best_costs = np.array(best_costs)
    
    return min_sol, best_costs, mean_costs, std_costs