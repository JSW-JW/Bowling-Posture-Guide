from pydantic import BaseModel
from typing import List, Dict

class AnalysisFeedback(BaseModel):
    torso: Dict[int, List[str]]
    foot: Dict[int, List[str]]
    stability: Dict[int, List[str]]

class AnalysisResult(BaseModel):
    feedback: AnalysisFeedback
    visualizations: Dict[str, str]
