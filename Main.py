import os
import sys
from typing import Dict, Any

from preprocessing import clean_text, tokenize_and_lemmatize, extract_named_entities
from skill_extractor import extract_skills, calculate_skill_match
from experience_extractor import extract_experience_years, calculate_experience_match
from feature_engineering import (
    compute_tfidf_similarity,
    compute_semantic_similarity,
    build_numeric_feature_vector,
)


def process_resume_matching(resume_text: str, jd_text: str, verbose: bool = True) -> Dict[str, Any]:
    # 0. Input Validation
    if not resume_text or not resume_text.strip():
        raise ValueError("Error: Resume text is empty or contains only whitespace.")
    if not jd_text or not jd_text.strip():
        raise ValueError("Error: Job description text is empty or contains only whitespace.")

    # Phase 3: NLP Processing
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = clean_text(jd_text)

    resume_nlp = tokenize_and_lemmatize(cleaned_resume)
    jd_nlp = tokenize_and_lemmatize(cleaned_jd)

    resume_entities = extract_named_entities(resume_text)
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    skill_score, matched_skills, missing_skills = calculate_skill_match(resume_skills, jd_skills)

    candidate_exp = extract_experience_years(resume_text)
    required_exp = extract_experience_years(jd_text)
    exp_score = calculate_experience_match(candidate_exp, required_exp)

    # Phase 4: Feature Engineering
    tfidf_sim = compute_tfidf_similarity(cleaned_resume, cleaned_jd)
    semantic_sim = compute_semantic_similarity(cleaned_resume, cleaned_jd)

    feature_vector_np, feature_vector_list = build_numeric_feature_vector(
        tfidf_sim=tfidf_sim,
        skill_match=skill_score,
        exp_match=exp_score,
        semantic_sim=semantic_sim,
        resume_skill_cnt=len(resume_skills),
        job_skill_cnt=len(jd_skills)
    )

    results = {
        "resume_skills": resume_skills,
        "job_skills": jd_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_score": skill_score,
        "candidate_experience_years": candidate_exp,
        "required_experience_years": required_exp,
        "experience_match_score": exp_score,
        "tfidf_similarity": tfidf_sim,
        "semantic_similarity": semantic_sim,
        "feature_vector": feature_vector_list,
        "feature_vector_numpy": feature_vector_np,
        "entities": resume_entities
    }

    if verbose:
        print_matching_report(results)

    return results


def print_matching_report(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print("      RESUME - JOB DESCRIPTION NLP & FEATURE MATCHING REPORT")
    print("=" * 65)

    print(f"\nResume Skills: {', '.join(results['resume_skills'])}")
    print(f"Job Description Skills: {', '.join(results['job_skills'])}")
    print(f"Matched Skills: {', '.join(results['matched_skills'])}")
    if results["missing_skills"]:
        print(f"Missing Skills: {', '.join(results['missing_skills'])}")

    print(f"\nSkill Match Score: {results['skill_match_score'] * 100:.1f}%")
    print(f"Candidate Experience: {results['candidate_experience_years']:.1f} years")
    print(f"Required Experience: {results['required_experience_years']:.1f} years")
    print(f"Experience Match Score: {results['experience_match_score'] * 100:.1f}%")

    print(f"\nTF-IDF Similarity: {results['tfidf_similarity']:.2f}")
    print(f"Semantic Similarity: {results['semantic_similarity']:.2f}")

    print("\nNumeric Feature Vector:")
    print(f"{results['feature_vector']}")
    print("\nFeature Vector Schema:")
    print("[tfidf_similarity, skill_match_score, experience_match_score, semantic_similarity, resume_skill_count, job_skill_count]")
    print("=" * 65 + "\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resume_path = os.path.join(base_dir, "sample_resume.txt")
    jd_path = os.path.join(base_dir, "sample_job_description.txt")

    with open(resume_path, "r", encoding="utf-8") as f:
        sample_resume = f.read()
    with open(jd_path, "r", encoding="utf-8") as f:
        sample_jd = f.read()

    process_resume_matching(sample_resume, sample_jd, verbose=True)


if __name__ == "__main__":
    main()
