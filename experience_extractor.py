
import re
from typing import Optional, Tuple

EXPERIENCE_PATTERNS = [
    # Format: "3+ years of experience", "2 years experience", "3+ yrs of experience"
    re.compile(r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+relevant|\s+work|\s+professional|\s+industry|\s+software)?\s+experience', re.IGNORECASE),

    # Format: "5 years in software development", "3 yrs in data science"
    re.compile(r'(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)\s+in\s+[a-z\s]{3,30}', re.IGNORECASE),

    # Format: "minimum 3 years", "at least 4 years", "over 5 years"
    re.compile(r'(?:minimum|min\.?|at least|over)\s+(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)', re.IGNORECASE),

    # Format: "experience: 3 years", "experience - 4+ yrs"
    re.compile(r'experience\s*[:\-]\s*(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)', re.IGNORECASE),

    # Format: "3 to 5 years", "3-5 years" -> extracts the minimum years
    re.compile(r'(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*(?:years?|yrs?)', re.IGNORECASE),
]


def extract_experience_years(text: str) -> float:
    """
    Extract total years of experience from text using regular expressions.
    """
    if not text or not text.strip():
        return 0.0

    found_years = []
    for pattern in EXPERIENCE_PATTERNS:
        matches = pattern.finditer(text)
        for match in matches:
            try:
                val = float(match.group(1))
                if 0.5 <= val <= 40.0:
                    found_years.append(val)
            except (ValueError, IndexError):
                continue

    if found_years:
        return max(found_years)

    # Heuristic fallback: Detect date intervals like "2021 - 2024" or "2021 - Present"
    date_range_pattern = re.compile(
        r'\b(20\d{2}|19\d{2})\s*(?:-|–|to)\s*(present|current|now|20\d{2})\b',
        re.IGNORECASE
    )
    ranges = date_range_pattern.findall(text)
    total_years = 0.0
    for start_yr, end_yr in ranges:
        try:
            s = int(start_yr)
            e = 2026 if end_yr.lower() in {"present", "current", "now"} else int(end_yr)
            if e >= s:
                total_years += (e - s)
        except ValueError:
            continue

    return min(float(total_years), 40.0) if total_years > 0 else 0.0


def calculate_experience_match(candidate_exp: float, required_exp: float) -> float:
    """
    Calculate the experience match score between candidate experience and JD requirements.
    - If required_exp == 0 -> 1.0 (100%)
    - If candidate_exp >= required_exp -> 1.0 (100%)
    - If candidate_exp < required_exp -> candidate_exp / required_exp
    """
    if required_exp <= 0.0:
        return 1.0 if candidate_exp > 0.0 else 0.8

    if candidate_exp >= required_exp:
        return 1.0

    score = candidate_exp / required_exp
    return round(max(0.0, min(1.0, score)), 4)
