import numpy as np


# Let us require that the payoff tensor is symmetric, that is, payoff_pl_i[k-th agent_type for player 1, ..., l-th agent_type for player i, ...] = payoff_pl_1[l-th agent_type for player 1, ..., k-th agent_type for player i, ...]. Hence, we only need to keep track of one tensor (of player 1).

class Population_Payoffs:
    def __init__(self, agent_types, payoff_tensor):
        """
        Initialize the payoffs of agent types vs other agent types, from the perspective of player 1 in the underlying game.
        
        Args:
            agent_types: List of agent types (e.g., ['chatgpt', 'claude', 'deepseek'])
            payoff_tensor: kD numpy array representing payoffs, where k is the number of agent types
        """
        assert all(dim == len(agent_types) for dim in payoff_tensor.shape), "Each dimension of payoff tensor must have size equal to number of agent types"
        self.agent_types = agent_types
        self.payoff_tensor = payoff_tensor

    def update_payoff_tensor(self, new_payoff_tensor):
        assert all(dim == len(self.agent_types) for dim in new_payoff_tensor.shape), "Each dimension of payoff tensor must have size equal to number of agent types"
        self.payoff_tensor = new_payoff_tensor

    def get_expected_payoffs(self, population):
        """
        Compute expected payoffs for player 1 when all other player types are sampled from the same population distribution.
        
        Args:
            population: numpy array of length len(agent_types)
        
        Returns:
            expected_payoffs: numpy array of length len(agent_types)
        """
        
        # Create einsum subscript string
        # payoff_tensor has payoff_tensor.ndim indices, we keep the first one (player 1) and contract the rest
        
        # Create einsum expression: 'abcd...,b,c,d,...->a'
        tensor_indices = ''.join(chr(ord('a') + i) for i in range(self.payoff_tensor.ndim))
        strategy_indices = tensor_indices[1:]  # indices for other players
        einsum_expr = tensor_indices + ',' + ','.join(strategy_indices) + '->' + 'a'
        
        # Input: tensor + (num_players-1) copies of the same mixed strategy
        inputs = [self.payoff_tensor] + [population] * (self.payoff_tensor.ndim - 1)
        
        return np.einsum(einsum_expr, *inputs)


class Discrete_Replicator_Dynamics:
    """
    Discrete-time replicator dynamics using exponential weight updates.
    
    This implements the update rule:
    x_i(t+1) = x_i(t) * exp(η * (f_i - f_avg)) / Z(t)
    
    where η is the learning rate and Z(t) is the normalization constant. For learning rate going to zero, this approaches the continuous-time replicator dynamics.
    """
    def __init__(self, pop_payoffs, payoffs_updating=False):
        self.population_payoffs = pop_payoffs
        if payoffs_updating:
            raise NotImplementedError("Payoff updates in between the dynamics is not implemented yet!")

    def replicator_dynamics_update(self, current_pop, fitness, avg_fitness, lr):
        """
        Args:
            current_dist: numpy array, current probability distribution over agent types
            fitness: numpy array, fitness of each agent type against current distribution  
            avg_fitness: float, current average performance
            t: int, current time step (must be > 0)
        
        Returns:
            numpy array, next step's probability distribution over agent types
        """
        weights = current_pop * np.exp(lr * (fitness - avg_fitness))
        next_pop = weights / np.sum(weights)
        
        return next_pop
    
    def run_mw_dynamics(self, initial_population="uniform", steps=1000, tol=1e-6, learning_rate={"method": "constant", "nu": 0.1}):
        """
        Run the multiplicative weights dynamics for a specified number of steps.
        """

        # Initialize learning rate function
        if learning_rate["method"] == "constant":
            lr_fct = lambda t: learning_rate["nu"]
        elif learning_rate["method"] == "sqrt":
            lr_fct = lambda t: learning_rate["nu"] / np.sqrt(t)
        else:
            raise ValueError("learning_rate method must be 'constant' or 'sqrt'")

        # Initialize population distribution
        if isinstance(initial_population, np.ndarray):
            assert len(initial_population) == len(self.population_payoffs.agent_types), "Initial population distribution must match number of agent types"
            assert np.all(initial_population >= 0), "Initial population distribution must be non-negative"
            population = initial_population
        elif initial_population == "random":
            population = np.random.exponential(scale=1.0, size=len(self.population_payoffs.agent_types))
        elif initial_population == "uniform":
            population = np.ones(len(self.population_payoffs.agent_types))
        else:
            raise ValueError("initial_population must be a numpy array or 'uniform'")
        population = population / np.sum(population)
        expected_payoffs = self.population_payoffs.get_expected_payoffs(population)
        avg_payoff = np.dot(population, expected_payoffs)    

        # Run the dynamics
        population_history = [population.copy()]
        payoff_history = [avg_payoff]
        for t in range(1, steps + 1):
            if np.all(expected_payoffs - avg_payoff < tol):
                status = "converged: approximate equilibrium reached"
                return population_history, payoff_history, status
            lr = lr_fct(t)
            population = self.replicator_dynamics_update(population, expected_payoffs, avg_payoff, lr)
            expected_payoffs = self.population_payoffs.get_expected_payoffs(population)
            avg_payoff = np.dot(population, expected_payoffs)
            population_history.append(population.copy())
            payoff_history.append(avg_payoff)
        
        status = "steps limit reached"
        return population_history, payoff_history, status