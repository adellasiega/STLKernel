import random
import torch
from torch.distributions import Distribution, Poisson
from typing import Optional, Type, Dict, List, Any
from torcheck import stl


class F0Depth:
    """
    Generator for random STL formulae following the syntax-tree random growing scheme.
    
    The generation process:
    1. Start from root node
    2. For each node, if depth_max is reached, make it an atomic predicate
    3. For internal nodes, sample operator type uniformly and recursively sample children
    4. Atomic predicates: xi ≤ theta or xi ≥ theta with random variable index and theta ~ N(0,1)
    5. Temporal operators: left and right bound sampled uniformly from {0,...,tmax}
    """
    
    def __init__(
        self,
        n_vars: int,
        t_max: int = 100,
        signal_length = 100,
        depth_max: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize the STL formula distribution generator.
        
        Args:
            n_vars: Number of signal variables/dimensions
            t_max: Maximum temporal bound for temporal operators
            signal_length: Number of time steps in the signal
            seed: Random seed for reproducibility
        """
        self.n_vars = n_vars
        self.t_max = t_max
        self.signal_length = signal_length
        self.depth_max = depth_max
        self.boolean_operators = [
            'And', 'Or', 'Not'
        ]
        self.temporal_operators = [
            'Globally', 'Eventually', 'Until'
        ]
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
    
    def _sample_formula(self, remaining_time: int, formula_depth: int, current_depth: int = 0):
        """
        Sample a random STL formula using the syntax-tree random growing scheme.
        
        Args:
            remaining_time: how many time steps are available in the signal            
        Returns:
            Node: A randomly generated STL formula tree
        """        
        if current_depth == formula_depth: # Leaf node
            return self._sample_atomic_predicate() 
        else:
            return self._sample_operator_node(remaining_time, formula_depth, current_depth+1)
    
    def _sample_atomic_predicate(self):
        """
        Sample an atomic predicate of the form xi ≤ theta or xi ≥ theta.
        
        Returns:
            Atom: An atomic predicate node
        """
        var_index = random.randint(0, self.n_vars - 1)
        threshold = torch.randn(1).item()
        lte = random.choice([True, False])
        
        return stl.Atom(var_index=var_index, threshold=threshold, lte=lte)
    
    def _sample_operator_node(self, remaining_time, formula_depth, current_depth):
        """
        Sample an operator node uniformly and recursively generate its children.
        
        Returns:
            Node: An operator node (And, Or, Not, Globally, Eventually, or Until)
        """
        allowed_operators = self.boolean_operators
        if remaining_time > 0 :
            allowed_operators += self.temporal_operators
        
        operator = random.choice(allowed_operators)

        if operator == 'And':
            left_child = self._sample_formula(remaining_time, formula_depth, current_depth)
            right_child = self._sample_formula(remaining_time, formula_depth, current_depth)
            return stl.And(left_child, right_child)
        
        elif operator == 'Or':
            left_child = self._sample_formula(remaining_time, formula_depth, current_depth)
            right_child = self._sample_formula(remaining_time, formula_depth, current_depth)
            return stl.Or(left_child, right_child)
        
        elif operator == 'Not':
            child = self._sample_formula(remaining_time, formula_depth, current_depth)
            return stl.Not(child)
        
        elif operator == 'Globally':
            b = random.randint(0, remaining_time)
            a = random.randint(0, b)
            child = self._sample_formula(remaining_time-b, formula_depth, current_depth)
            return stl.Globally(
                child,
                unbound=False,
                left_time_bound=a,
                right_time_bound=b
            )
        
        elif operator == 'Eventually':
            b = random.randint(0, remaining_time)
            a = random.randint(0, b)
            child = self._sample_formula(remaining_time-b, formula_depth, current_depth)
            return stl.Eventually(
                child,
                unbound=False,
                left_time_bound=a,
                right_time_bound=b
            )
        
        else: # Until
            b = random.randint(0, remaining_time)
            a = random.randint(0, b)
            left_child = self._sample_formula(remaining_time-b, formula_depth, current_depth)
            right_child = self._sample_formula(remaining_time-b, formula_depth, current_depth)
            return stl.Until(
                left_child,
                right_child,
                unbound=False,
                left_time_bound=a,
                right_time_bound=b
            )
    
    def sample(self, depths: List) -> list:
            """
            Samples a formula for each depth sepcified.
            Args: 
                depths: list of integres, each element is the depth of the formula to be sampled
            Returns:
                list of formulae 
            """
            remaining_time = min(self.t_max, self.signal_length - 1)
            return [
                self._sample_formula(remaining_time, formula_depth=min(d, self.depth_max))
                for d in depths
            ]
    
    def get_formula_depth(self, formula, current_depth: int = 0) -> int:
        """
        Compute the current_depth of the formula tree.
        
        Args:
            formula: STL formula node
            current_depth: Current current_depth in recursion
            
        Returns:
            int: Maximum current_depth of the formula tree
        """
        if isinstance(formula, stl.Atom):
            return current_depth
        
        elif isinstance(formula, stl.Not):
            return self.get_formula_depth(formula.child, current_depth + 1)
        
        elif isinstance(formula, (stl.Globally, stl.Eventually)):
            return self.get_formula_depth(formula.child, current_depth + 1)
        
        elif isinstance(formula, (stl.And, stl.Or)):
            left_depth = self.get_formula_depth(formula.left_child, current_depth + 1)
            right_depth = self.get_formula_depth(formula.right_child, current_depth + 1)
            return max(left_depth, right_depth)
        
        elif isinstance(formula, stl.Until):
            left_depth = self.get_formula_depth(formula.left_child, current_depth + 1)
            right_depth = self.get_formula_depth(formula.right_child, current_depth + 1)
            return max(left_depth, right_depth)
        
        else:
            return current_depth
        

class F0:
    """
    Generator for random STL formulae following the syntax-tree random growing scheme.
    
    The generation process:
    1. Start from root (forced to be an operator node)
    2. For each node, with probability p_leaf make it an atomic predicate
    3. For internal nodes, sample operator type uniformly and recursively sample children
    4. Atomic predicates: xi ≤ theta or xi ≥ theta with random variable index and theta ~ N(0,1)
    5. Temporal operators: right bound uniform from {1,...,tmax}, left bound = 0
    """
    
    def __init__(
        self,
        n_vars: int,
        p_leaf: float = 0.5,
        t_max: int = 10,
        operators: Optional[list] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the STL formula distribution generator.
        
        Args:
            n_vars: Number of signal variables/dimensions
            p_leaf: Probability of creating a leaf (atomic predicate) node
            t_max: Maximum temporal bound for temporal operators
            operators: List of operator types to use. If None, uses all available.
            seed: Random seed for reproducibility
        """
        self.n_vars = n_vars
        self.p_leaf = p_leaf
        self.t_max = t_max
        
        # Define available operators
        if operators is None:
            self.operators = [
                'And', 'Or', 'Not',
                'Globally', 'Eventually', 'Until'
            ]
        else:
            self.operators = operators
        
        self.unary_operators = ['Not', 'Globally', 'Eventually']
        self.binary_operators = ['And', 'Or', 'Until']
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
    
    def sample_formula(self, is_root: bool = True):
        """
        Sample a random STL formula using the syntax-tree random growing scheme.
        
        Args:
            is_root: Whether this is the root node (forced to be an operator)
            
        Returns:
            Node: A randomly generated STL formula tree
        """
        # Root must be an operator, other nodes can be leaves with probability p_leaf
        if is_root or random.random() > self.p_leaf:
            return self._sample_operator_node()
        else:
            return self._sample_atomic_predicate()
    
    def _sample_atomic_predicate(self):
        """
        Sample an atomic predicate of the form xi ≤ theta or xi ≥ theta.
        
        Returns:
            Atom: An atomic predicate node
        """
        # Sample variable index uniformly from available dimensions
        var_index = random.randint(0, self.n_vars - 1)
        
        # Sample threshold from standard Gaussian N(0, 1)
        threshold = torch.randn(1).item()
        
        # Sample inequality type (lte: True for ≤, False for ≥)
        lte = random.choice([True, False])
        
        return stl.Atom(var_index=var_index, threshold=threshold, lte=lte)
    
    def _sample_operator_node(self):
        """
        Sample an operator node uniformly and recursively generate its children.
        
        Returns:
            Node: An operator node (And, Or, Not, Globally, Eventually, or Until)
        """
        # Sample operator type uniformly from available operators
        operator = random.choice(self.operators)
        
        if operator == 'And':
            left_child = self.sample_formula(is_root=False)
            right_child = self.sample_formula(is_root=False)
            return stl.And(left_child, right_child)
        
        elif operator == 'Or':
            left_child = self.sample_formula(is_root=False)
            right_child = self.sample_formula(is_root=False)
            return stl.Or(left_child, right_child)
        
        elif operator == 'Not':
            child = self.sample_formula(is_root=False)
            return stl.Not(child)
        
        elif operator == 'Globally':
            child = self.sample_formula(is_root=False)
            # Left bound = 0, right bound uniform from {1, ..., tmax}
            right_bound = random.randint(1, self.t_max)
            return stl.Globally(
                child,
                unbound=False,
                left_time_bound=0,
                right_time_bound=right_bound
            )
        
        elif operator == 'Eventually':
            child = self.sample_formula(is_root=False)
            # Left bound = 0, right bound uniform from {1, ..., tmax}
            right_bound = random.randint(1, self.t_max)
            return stl.Eventually(
                child,
                unbound=False,
                left_time_bound=0,
                right_time_bound=right_bound
            )
        
        elif operator == 'Until':
            left_child = self.sample_formula(is_root=False)
            right_child = self.sample_formula(is_root=False)
            # Left bound = 0, right bound uniform from {1, ..., tmax}
            right_bound = random.randint(1, self.t_max)
            return stl.Until(
                left_child,
                right_child,
                unbound=False,
                left_time_bound=0,
                right_time_bound=right_bound
            )
        
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def sample(self, n_formulae: int) -> list:
        """
        Sample multiple random STL formulae.
        
        Args:
            n_formulae: Number of formulae to generate
            
        Returns:
            list: List of randomly generated STL formula trees
        """
        return [self.sample_formula() for _ in range(n_formulae)]
    
    def get_formula_depth(self, formula, current_depth: int = 0) -> int:
        """
        Compute the depth of the formula tree.
        
        Args:
            formula: STL formula node
            current_depth: Current depth in recursion
            
        Returns:
            int: Maximum depth of the formula tree
        """
        if isinstance(formula, stl.Atom):
            return current_depth
        
        elif isinstance(formula, stl.Not):
            return self.get_formula_depth(formula.child, current_depth + 1)
        
        elif isinstance(formula, (stl.Globally, stl.Eventually)):
            return self.get_formula_depth(formula.child, current_depth + 1)
        
        elif isinstance(formula, (stl.And, stl.Or)):
            left_depth = self.get_formula_depth(formula.left_child, current_depth + 1)
            right_depth = self.get_formula_depth(formula.right_child, current_depth + 1)
            return max(left_depth, right_depth)
        
        elif isinstance(formula, stl.Until):
            left_depth = self.get_formula_depth(formula.left_child, current_depth + 1)
            right_depth = self.get_formula_depth(formula.right_child, current_depth + 1)
            return max(left_depth, right_depth)
        
        else:
            return current_depth