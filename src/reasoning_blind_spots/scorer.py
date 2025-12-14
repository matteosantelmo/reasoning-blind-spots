from inspect_ai.scorer import model_graded_qa, Scorer
from inspect_ai.model import get_model

def get_verifier(model_name: str = "google/gemini-2.5-flash-lite") -> Scorer:
    """
    Returns a model-graded scorer (verifier) using the specified model.
    """
    model = get_model(model=model_name, role="grader")
    return model_graded_qa(
        model=model, 
        template="""
        You are an expert grader evaluating AI model outputs.
        
        Question: {question}
        
        Reference Solution / Evaluation Criteria:
        {criterion}
        
        Candidate Answer:
        {answer}
        
        Based on the Reference Solution and Criteria, determine if the Candidate Answer is correct.
        If the solution provides specific values, check for them.
        If the solution provides behavioral rules, check if they are followed.
        
        Return 'GRADE: C' for Correct or 'GRADE: I' for Incorrect.
        """
    )
