import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import json
import os
import asyncio
import re
from reasoning_blind_spots.grader import get_grader

"""
Evaluate a set of graders for solver given the human grading of the solver
Returns the binary metric for the accuracy of the evaluation
"""
def compute_metrics(grader_scores,human_scores):
    l = len(grader_scores)
    TP=FP=TN=FN=0
    for i in range(l):
        if grader_scores[i]=="C" and human_scores[i]=="C":
            TP+=1
        elif grader_scores[i]=="C" and human_scores[i]=="I":
            FP+=1
        elif grader_scores[i]=="I" and human_scores[i]=="C":
            FN+=1
        elif grader_scores[i]=="I" and human_scores[i]=="I":
            TN +=1

    accuracy = (TP+TN)/l
    precision = TP/(TP+FP) if (TP+FP)>0 else 0
    recall = TP/(TP+FN) if (TP+FN)>0 else 0
    FPR = FP/(FP+TN) if (FP+TN)>0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "FPR": FPR
    }
    
"""
The evaluation.jsonl file contains the prompt, groundtruth answer, solver answer, solver model and human grading. 
This is passed with the Config for the grader. It returns the evaluation of the grader.
"""
async def grader_validation(cfg: DictConfig, val_path):
    grader_scores = []
    human_scores = []
    grader = get_grader(cfg.grader,str_input=True)
    with open(val_path,"r") as f:
        for line in f:
            sample = json.loads(line)
            prompt = sample["prompt"]
            solver_answer = sample["solver_answer"]
            solution = sample["solution"]
            human_score = sample["human_score"]
            human_scores.append(human_score)
            score = await grader(prompt,solver_answer,solution)
            grader_scores.append(score.value)

    return compute_metrics(grader_scores,human_scores)

