"""
OpenEnv RFC 004 Compliant Rubric System for Support Triage.

Separates intermediate process rewards (tool use, valid syntax) 
from explicit outcome rewards (final classification accuracy).
"""

from typing import Dict, Any, Tuple, Optional
from .graders import final_grader


class ProcessRubric:
    """Evaluates intermediate steps before the episode concludes."""
    
    def score(self, action: Any, error: Optional[str], state: Any, submission: Dict[str, Any], task_name: str) -> float:
        process_reward = 0.0
        
        if error:
            # Penalize malformed actions or empty submissions
            process_reward -= 0.05
            
        if action.tool_call:
            # Reward valid tool exploration highly
            if action.tool_call in ["check_customer_tier", "check_system_status"]:
                process_reward += 0.1
                
        # Action Consequences: Penalize skipping required fields
        if not action.tool_call and not error:
            if task_name == "full_resolution" and "reply" not in submission and action.reply is None:
                process_reward -= 0.1
            if task_name == "escalation_detection" and "escalate" not in submission and action.escalate is None:
                process_reward -= 0.1
                
        return min(1.0, max(-1.0, process_reward))


class OutcomeRubric:
    """Evaluates final task completion."""
    
    def score(self, task_name: str, submission: Dict[str, Any], ground_truth: Dict[str, Any], state: Any) -> float:
        is_probe = ground_truth.get("is_probe", False)
        if is_probe:
            return 1.0
            
        reward = final_grader(task_name, submission, ground_truth)
        
        # Penalize performing final action without investigation if information was needed
        needs_tier = ground_truth.get("plan") in ["Pro", "Starter", "Enterprise"]
        if needs_tier and "check_customer_tier" not in state.used_tools:
            # Strong penalty for guessing blind
            reward -= 0.3
            
        return max(0.01, min(0.99, reward))


class SupportTriageRubric:
    """Composite rubric for Support Triage tasks following OpenEnv RFC 004."""
    
    def __init__(self):
        self.process = ProcessRubric()
        self.outcome = OutcomeRubric()

    def score_step(
        self, 
        done: bool, 
        action: Any, 
        error: Optional[str], 
        task_name: str, 
        submission: Dict[str, Any], 
        ground_truth: Dict[str, Any],
        state: Any
    ) -> float:
        """
        Returns the temporally-discounted or discrete RFC 004 score for the current step.
        """
        if done:
            return self.outcome.score(task_name, submission, ground_truth, state)
        
        return self.process.score(action, error, state, submission, task_name)
