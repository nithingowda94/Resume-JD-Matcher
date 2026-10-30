
import re
from typing import List, Dict, Set, Tuple

# Predefined skill dictionary mapping skill variants/synonyms to canonical names
SKILL_DICTIONARY: Dict[str, str] = {
    # Programming Languages
    "python": "Python",
    "python3": "Python",
    "py": "Python",
    "java": "Java",
    "core java": "Java",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    ".net": ".NET",
    "dotnet": ".NET",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "r": "R",
    "golang": "Go",
    "go": "Go",
    "rust": "Rust",
    "ruby": "Ruby",
    "php": "PHP",
    "sql": "SQL",
    "pl/sql": "SQL",
    "bash": "Bash",
    "shell": "Bash",

    # Web Development
    "html": "HTML",
    "html5": "HTML",
    "css": "CSS",
    "css3": "CSS",
    "react": "React",
    "react.js": "React",
    "reactjs": "React",
    "angular": "Angular",
    "vue": "Vue",
    "node": "Node.js",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "express": "Express.js",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "spring": "Spring Boot",
    "spring boot": "Spring Boot",
    "rest": "REST API",
    "rest api": "REST API",
    "restful api": "REST API",
    "graphql": "GraphQL",

    # Machine Learning & AI
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "deep learning": "Deep Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "computer vision": "Computer Vision",
    "cv": "Computer Vision",
    "data science": "Data Science",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "tensorflow": "TensorFlow",
    "tf": "TensorFlow",
    "pytorch": "PyTorch",
    "torch": "PyTorch",
    "keras": "Keras",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "spacy": "spaCy",
    "nltk": "NLTK",
    "llm": "LLM",
    "generative ai": "Generative AI",

    # Databases & Big Data
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "spark": "Spark",
    "pyspark": "PySpark",
    "hadoop": "Hadoop",
    "kafka": "Kafka",

    # Cloud & DevOps
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "aws": "AWS",
    "amazon web services": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "git": "Git",
    "github": "Git",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "jenkins": "Jenkins",
    "linux": "Linux",
}


def extract_skills(text: str) -> List[str]:
    """
    Extract technical and professional skills from text.
    Matches aliases and maps them to their canonical representation.
    """
    if not text or not text.strip():
        return []

    found_skills: Set[str] = set()
    sorted_aliases = sorted(SKILL_DICTIONARY.keys(), key=lambda x: len(x), reverse=True)

    for alias in sorted_aliases:
        canonical_name = SKILL_DICTIONARY[alias]
        if canonical_name in found_skills:
            continue

        if alias in {"c++", "c#", ".net"}:
            pattern = rf'(?:^|[\s,;:(/]){re.escape(alias)}(?:$|[\s,;:)/])'
        elif "/" in alias:
            pattern = rf'(?:^|[\s,;:(]){re.escape(alias)}(?:$|[\s,;:)/])'
        else:
            pattern = rf'\b{re.escape(alias)}\b'

        if re.search(pattern, text, re.IGNORECASE):
            found_skills.add(canonical_name)

    return sorted(list(found_skills))


def calculate_skill_match(
    resume_skills: List[str],
    job_skills: List[str]
) -> Tuple[float, List[str], List[str]]:
    """
    Calculate the skill match percentage between candidate skills and job requirements.
    Formula: Matched Skills / Total Job Required Skills
    """
    resume_set = set(resume_skills)
    job_set = set(job_skills)

    matched_skills = sorted(list(job_set.intersection(resume_set)))
    missing_skills = sorted(list(job_set - resume_set))

    if not job_set:
        match_score = 1.0 if resume_set else 0.0
    else:
        match_score = len(matched_skills) / len(job_set)

    return round(match_score, 4), matched_skills, missing_skills
