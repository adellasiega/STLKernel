import random
import torch
from torch.distributions import Distribution, Poisson
from typing import Optional, Type, Dict, List, Any
from torcheck import stl


class F0:
    def __init__(
        self,
        n_vars: int,
        v_min: torch.Tensor,
        v_max: torch.Tensor,
        t_max: int = 100,
        depth_max: int = 5,
        p_base: float = 0.05,  # 5% base chance to stop at any level
        seed: Optional[int] = None
    ):
        self.n_vars = n_vars
        self.v_min = v_min
        self.v_max = v_max
        self.t_max = t_max
        self.depth_max = depth_max
        self.p_base = p_base
        
        self.boolean_operators = ['And', 'Or', 'Not']
        self.temporal_operators = ['Globally', 'Eventually', 'Until']
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def _get_term_probability(self, current_depth: int) -> float:
        """
        Calculates the probability to stop and create an Atom.
        Even at depth 0, there is a 'p_base' chance to terminate.
        """
        # Linear growth from p_base to 1.0
        growth = current_depth / self.depth_max
        return min(1.0, self.p_base + (1.0 - self.p_base) * growth)

    def _sample_formula(self, remaining_time: int, current_depth: int = 0):
        # Decide termination
        if current_depth >= self.depth_max or random.random() < self._get_term_probability(current_depth):
            return self._sample_atomic_predicate()
        
        return self._sample_operator_node(remaining_time, current_depth)

    def _sample_atomic_predicate(self):
        var_idx = random.randint(0, self.n_vars - 1)
        v_low = self.v_min[var_idx].item()
        v_high = self.v_max[var_idx].item()
        
        # Use a uniform distribution over the actual data range
        threshold = random.uniform(v_low, v_high)
        lte = random.choice([True, False])
        
        return stl.Atom(var_index=var_idx, threshold=threshold, lte=lte)

    def _sample_operator_node(self, remaining_time: int, current_depth: int):
        allowed = list(self.boolean_operators)
        # Check if we have enough time left for temporal logic
        if remaining_time > 2:
            allowed += self.temporal_operators
        
        op = random.choice(allowed)

        if op in ['And', 'Or']:
            left = self._sample_formula(remaining_time, current_depth + 1)
            right = self._sample_formula(remaining_time, current_depth + 1)
            return stl.And(left, right) if op == 'And' else stl.Or(left, right)
        
        elif op == 'Not':
            return stl.Not(self._sample_formula(remaining_time, current_depth + 1))
        
        elif op in ['Globally', 'Eventually']:
            b = random.randint(1, remaining_time - 1)
            a = random.randint(0, b - 1)
            child = self._sample_formula(remaining_time - b, current_depth + 1)
            
            OpClass = stl.Globally if op == 'Globally' else stl.Eventually
            return OpClass(child, unbound=False, left_time_bound=a, right_time_bound=b)
        
        else: # Until
            b = random.randint(1, remaining_time - 1)
            a = random.randint(0, b - 1)
            # Remaining time is reduced for both children to ensure validity
            left = self._sample_formula(remaining_time - b, current_depth + 1)
            right = self._sample_formula(remaining_time - b, current_depth + 1)
            return stl.Until(left, right, unbound=False, left_time_bound=a, right_time_bound=b)

    def sample(self, n_formulae: int) -> List:
        """Returns a list of n_formulae with highly varied structures."""
        initial_time = self.t_max - 1 
        return [self._sample_formula(initial_time, current_depth=0) for _ in range(n_formulae)]