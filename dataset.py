"""Tiny knowledge corpus + labeled eval set (with categories and rubrics).

Self-contained (does not import ai-learn-08), but the corpus style mirrors it.
Each eval item has:
  - question / gold (short reference answer) / aliases (accepted variants)
  - category (for per-category breakdown)
  - rubric: required keywords (all must appear), optional bonus keywords,
    forbidden keywords (hallucination / wrong-fact markers) and a max length.
"""
from __future__ import annotations

from typing import Dict, List

DOCUMENTS: List[Dict[str, str]] = [
    {"id": "py1", "topic": "python", "text": "Python was created by Guido van Rossum. It emphasizes readability and uses significant indentation."},
    {"id": "py2", "topic": "python", "text": "A Python list is an ordered mutable sequence. You add items to a list with the append method."},
    {"id": "py3", "topic": "python", "text": "Python dictionaries map keys to values. Key lookup is fast because dictionaries use a hash table."},
    {"id": "py4", "topic": "python", "text": "Virtual environments isolate package installs per project. You create one with python -m venv."},
    {"id": "py5", "topic": "python", "text": "A tuple is an immutable ordered sequence. Tuples cannot be changed after creation."},
    {"id": "ml1", "topic": "ml", "text": "Overfitting happens when a model memorizes training noise. It performs well on training data but poorly on unseen data."},
    {"id": "ml2", "topic": "ml", "text": "Gradient descent updates parameters in the direction of the negative gradient. The learning rate controls the step size."},
    {"id": "ml3", "topic": "ml", "text": "K-fold cross-validation splits data into k folds. Each fold is used once as the validation set."},
    {"id": "ml4", "topic": "ml", "text": "Regularization such as L2 weight decay penalizes large weights. It reduces overfitting."},
    {"id": "ml5", "topic": "ml", "text": "Precision is the fraction of predicted positives that are correct. Recall is the fraction of actual positives that are found."},
    {"id": "nlp1", "topic": "nlp", "text": "Byte pair encoding builds a subword vocabulary by merging frequent symbol pairs."},
    {"id": "nlp2", "topic": "nlp", "text": "Transformers use self-attention to mix information across token positions."},
    {"id": "nlp3", "topic": "nlp", "text": "Sinusoidal positional encodings use sine and cosine waves of different frequencies."},
    {"id": "nlp4", "topic": "nlp", "text": "Temperature scaling divides logits before softmax. A low temperature makes sampling more greedy."},
    {"id": "nlp5", "topic": "nlp", "text": "Retrieval-augmented generation retrieves relevant documents and conditions the answer on them."},
    {"id": "sci1", "topic": "science", "text": "Photosynthesis mainly occurs in the chloroplasts of plant leaves."},
    {"id": "sci2", "topic": "science", "text": "Mitochondria are often called the powerhouse of the cell because they produce ATP."},
    {"id": "sci3", "topic": "science", "text": "Water boils at 100 degrees Celsius at sea level."},
    {"id": "sci4", "topic": "science", "text": "The speed of light in vacuum is about 300000 kilometers per second."},
    {"id": "sci5", "topic": "science", "text": "A cold front often brings thunderstorms followed by cooler, drier air."},
]

EVAL_SET: List[Dict[str, object]] = [
    # python
    {"id": "e01", "category": "python", "question": "Who created Python?", "gold": "Guido van Rossum", "aliases": ["van Rossum"],
     "rubric": {"required": ["guido", "rossum"], "bonus": ["python"], "forbidden": ["java"], "max_words": 30}},
    {"id": "e02", "category": "python", "question": "Which method adds an item to a Python list?", "gold": "append", "aliases": ["the append method"],
     "rubric": {"required": ["append"], "bonus": ["list"], "forbidden": [], "max_words": 30}},
    {"id": "e03", "category": "python", "question": "Why is dictionary key lookup fast in Python?", "gold": "dictionaries use a hash table", "aliases": ["hash table"],
     "rubric": {"required": ["hash"], "bonus": ["table"], "forbidden": [], "max_words": 30}},
    {"id": "e04", "category": "python", "question": "How do you create a virtual environment?", "gold": "python -m venv", "aliases": ["venv"],
     "rubric": {"required": ["venv"], "bonus": ["python"], "forbidden": [], "max_words": 30}},
    {"id": "e05", "category": "python", "question": "Can a tuple be changed after creation?", "gold": "no, tuples are immutable", "aliases": ["no", "immutable"],
     "rubric": {"required": ["immutable"], "bonus": ["cannot"], "forbidden": [], "max_words": 30}},
    {"id": "e06", "category": "python", "question": "What language feature does Python use for code blocks?", "gold": "significant indentation", "aliases": ["indentation"],
     "rubric": {"required": ["indentation"], "bonus": ["significant"], "forbidden": ["braces"], "max_words": 30}},
    # ml
    {"id": "e07", "category": "ml", "question": "What is overfitting?", "gold": "a model memorizes training noise", "aliases": ["memorizes training noise"],
     "rubric": {"required": ["memorizes", "noise"], "bonus": ["unseen"], "forbidden": [], "max_words": 30}},
    {"id": "e08", "category": "ml", "question": "What controls the step size in gradient descent?", "gold": "the learning rate", "aliases": ["learning rate"],
     "rubric": {"required": ["learning", "rate"], "bonus": ["step"], "forbidden": [], "max_words": 30}},
    {"id": "e09", "category": "ml", "question": "How many times is each fold used for validation in k-fold cross-validation?", "gold": "once", "aliases": ["one time"],
     "rubric": {"required": ["once"], "bonus": ["validation"], "forbidden": [], "max_words": 30}},
    {"id": "e10", "category": "ml", "question": "What does L2 weight decay penalize?", "gold": "large weights", "aliases": ["big weights"],
     "rubric": {"required": ["large", "weights"], "bonus": ["overfitting"], "forbidden": [], "max_words": 30}},
    {"id": "e11", "category": "ml", "question": "What is recall?", "gold": "the fraction of actual positives that are found", "aliases": ["fraction of actual positives found"],
     "rubric": {"required": ["actual", "positives"], "bonus": ["found"], "forbidden": [], "max_words": 30}},
    {"id": "e12", "category": "ml", "question": "What helps a model generalize instead of memorizing noise?", "gold": "regularization", "aliases": ["weight decay", "L2 regularization"],
     "rubric": {"required": ["regularization"], "bonus": ["weight"], "forbidden": [], "max_words": 30}},
    # nlp
    {"id": "e13", "category": "nlp", "question": "How does byte pair encoding build its vocabulary?", "gold": "by merging frequent symbol pairs", "aliases": ["merging frequent pairs"],
     "rubric": {"required": ["merging", "pairs"], "bonus": ["frequent"], "forbidden": [], "max_words": 30}},
    {"id": "e14", "category": "nlp", "question": "What mechanism do transformers use to mix information across positions?", "gold": "self-attention", "aliases": ["attention", "self attention"],
     "rubric": {"required": ["attention"], "bonus": ["self"], "forbidden": ["recurrence"], "max_words": 30}},
    {"id": "e15", "category": "nlp", "question": "Which functions do sinusoidal positional encodings use?", "gold": "sine and cosine", "aliases": ["sine and cosine waves"],
     "rubric": {"required": ["sine", "cosine"], "bonus": ["frequencies"], "forbidden": [], "max_words": 30}},
    {"id": "e16", "category": "nlp", "question": "What does a low temperature do to sampling?", "gold": "makes sampling more greedy", "aliases": ["more greedy"],
     "rubric": {"required": ["greedy"], "bonus": ["low"], "forbidden": [], "max_words": 30}},
    {"id": "e17", "category": "nlp", "question": "What does RAG condition the answer on?", "gold": "retrieved relevant documents", "aliases": ["relevant documents", "retrieved documents"],
     "rubric": {"required": ["documents"], "bonus": ["retrieves"], "forbidden": [], "max_words": 30}},
    {"id": "e18", "category": "nlp", "question": "What splits rare words into smaller units for a tokenizer?", "gold": "subword vocabulary via byte pair encoding", "aliases": ["byte pair encoding", "BPE"],
     "rubric": {"required": ["subword"], "bonus": ["byte"], "forbidden": [], "max_words": 30}},
    # science
    {"id": "e19", "category": "science", "question": "Where does photosynthesis mainly occur?", "gold": "in the chloroplasts", "aliases": ["chloroplasts"],
     "rubric": {"required": ["chloroplasts"], "bonus": ["leaves"], "forbidden": [], "max_words": 30}},
    {"id": "e20", "category": "science", "question": "What are mitochondria called?", "gold": "the powerhouse of the cell", "aliases": ["powerhouse of the cell"],
     "rubric": {"required": ["powerhouse"], "bonus": ["atp"], "forbidden": [], "max_words": 30}},
    {"id": "e21", "category": "science", "question": "At what temperature does water boil at sea level?", "gold": "100 degrees Celsius", "aliases": ["100 C", "100"],
     "rubric": {"required": ["100"], "bonus": ["celsius"], "forbidden": [], "max_words": 30}},
    {"id": "e22", "category": "science", "question": "How fast does light travel in vacuum?", "gold": "about 300000 kilometers per second", "aliases": ["300000 km/s"],
     "rubric": {"required": ["300000"], "bonus": ["kilometers"], "forbidden": [], "max_words": 30}},
    {"id": "e23", "category": "science", "question": "What weather follows a cold front?", "gold": "thunderstorms then cooler drier air", "aliases": ["cooler drier air"],
     "rubric": {"required": ["cooler"], "bonus": ["thunderstorms"], "forbidden": [], "max_words": 30}},
    {"id": "e24", "category": "science", "question": "Which organelle produces ATP?", "gold": "mitochondria", "aliases": ["the mitochondria"],
     "rubric": {"required": ["mitochondria"], "bonus": ["atp"], "forbidden": [], "max_words": 30}},
]
