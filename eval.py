"""
Evaluation & Test Suite
=======================
Validates that the Policy Manager Assistant produces clear, concise, and accurate responses.

"""
import json
import time
import re
import logging
from dataclasses import dataclass

from ingest import run_ingestion
from retriever import PolicyRetriever
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Test Cases 


@dataclass
class TestCase:
    name: str
    question: str
    expected_keywords: list[str]      # Must be present (case-insensitive)
    expected_missing: list[str] = None  # Must NOT be present
    max_words: int = 150               # Conciseness check
    should_cite: bool = True           # Should reference source docs
    should_refuse: bool = False        # Should decline to answer


TEST_CASES = [
    #  Accuracy: factual retrieval 
    TestCase(
        name="home_water_damage",
        question="What does my home policy cover for water damage?",
        expected_keywords=["burst pipes", "500"],  # EUR 500 deductible, burst pipes covered
        max_words=120,
    ),
    TestCase(
        name="home_flood_exclusion",
        question="Is flood damage covered under my home insurance?",
        expected_keywords=["not covered", "flood"],  # Flood is explicitly excluded
        expected_missing=["covered under this policy"],  # Should NOT say it's covered
        max_words=100,
    ),
    TestCase(
        name="auto_deductible",
        question="What is the deductible for a car collision?",
        expected_keywords=["500"],  # EUR 500 collision deductible
        max_words=80,
    ),
    TestCase(
        name="health_dental",
        question="What dental coverage do I have?",
        expected_keywords=["cleaning", "5,000"],  # 2 cleanings/year, EUR 5,000 limit
        max_words=150,
    ),
    TestCase(
        name="auto_glass",
        question="Is windshield repair covered for my car?",
        expected_keywords=["no deductible"],  # Glass breakage has no deductible
        max_words=80,
    ),

    #  Conciseness: process/workflow questions 
    TestCase(
        name="home_claims_process",
        question="How do I file a home insurance claim?",
        expected_keywords=["72 hours", "photograph"],  # Report within 72h, document with photos
        max_words=200,  # Process answers can be longer but still bounded
    ),
    TestCase(
        name="auto_renewal",
        question="How do I renew my auto policy?",
        expected_keywords=["automatically", "30 days"],  # Auto-renews, 30 days notice to cancel
        max_words=150,
    ),

    #  Grounding: cross-document comparison 
    TestCase(
        name="compare_deductibles",
        question="Compare the deductibles across all my policies",
        expected_keywords=["home", "auto", "health"],  # Should reference all three
        max_words=250,
    ),

    #  Grounding: refusal (hallucination prevention) 
    TestCase(
        name="crypto_theft_refusal",
        question="Does my policy cover cryptocurrency theft?",
        expected_keywords=["don't have that information"],
        should_refuse=True,
        max_words=60,
    ),
    TestCase(
        name="off_topic_refusal",
        question="What's the best restaurant in Zurich?",
        expected_keywords=[],  # No specific keywords needed
        should_refuse=True,
        max_words=60,
    ),

    # Conciseness: simple factual question 
    TestCase(
        name="home_coverage_limit",
        question="What is the total dwelling coverage limit for my home policy?",
        expected_keywords=["500,000"],  # EUR 500,000
        max_words=60,
    ),
]

# Filler phrases that indicate verbosity 
FILLER_PHRASES = [
    "great question",
    "that's a great question",
    "based on the provided context",
    "based on the information provided",
    "according to the context",
    "let me help you with that",
    "i'd be happy to help",
    "certainly",
    "absolutely",
    "of course",
]


#  Evaluation Functions

def check_accuracy(answer: str, test: TestCase) -> tuple[bool, list[str]]:
    """Check if expected keywords are present in the answer."""
    answer_lower = answer.lower()
    missing = []
    for keyword in test.expected_keywords:
        if keyword.lower() not in answer_lower:
            missing.append(keyword)
    return len(missing) == 0, missing


def check_no_hallucination(answer: str, test: TestCase) -> tuple[bool, list[str]]:
    """Check that forbidden phrases are NOT in the answer."""
    if not test.expected_missing:
        return True, []
    answer_lower = answer.lower()
    found = []
    for phrase in test.expected_missing:
        if phrase.lower() in answer_lower:
            found.append(phrase)
    return len(found) == 0, found


def check_conciseness(answer: str, test: TestCase) -> tuple[bool, int]:
    """Check word count is within bounds."""
    word_count = len(answer.split())
    return word_count <= test.max_words, word_count


def check_no_filler(answer: str) -> tuple[bool, list[str]]:
    """Check for verbose filler phrases."""
    answer_lower = answer.lower()
    found = [f for f in FILLER_PHRASES if f in answer_lower]
    return len(found) == 0, found


def check_citation(answer: str, test: TestCase) -> bool:
    """Check if the answer cites a source document."""
    if not test.should_cite:
        return True
    # Look for citation patterns like [Home Policy - ...] or [Source: ...]
    has_brackets = bool(re.search(r'\[.*?(Policy|Source|Section).*?\]', answer, re.IGNORECASE))
    has_reference = any(term in answer.lower() for term in
                        ["home policy", "auto policy", "health policy",
                         "coverage section", "exclusion", "source"])
    return has_brackets or has_reference


def check_refusal(answer: str, test: TestCase) -> bool:
    """Check if the system correctly refuses to answer."""
    if not test.should_refuse:
        return True
    refusal_indicators = [
        "don't have that information",
        "not covered in",
        "not available in",
        "outside",
        "cannot help with",
        "contact your zurich agent",
        "not related to",
    ]
    answer_lower = answer.lower()
    return any(ind in answer_lower for ind in refusal_indicators)


# Main Evaluation Runner 

def run_evaluation():
    """Run all test cases and produce a scored report."""
    print("=" * 70)
    print("POLICY MANAGER ASSISTANT — EVALUATION SUITE")
    print("=" * 70)
    print(f"Mode: {config.MODE}")
    print(f"Test cases: {len(TEST_CASES)}")
    print()

    # Step 1: Ingest documents
    print("Ingesting documents...")
    run_ingestion()
    print()

    # Step 2: Initialize retriever
    retriever = PolicyRetriever()
    print()

    # Step 3: Run test cases
    results = []
    total_score = 0
    max_score = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"─ Test {i}/{len(TEST_CASES)}: {test.name} ─")
        print(f"Q: {test.question}")

        start = time.time()
        result = retriever.query(test.question)
        elapsed = time.time() - start
        answer = result["answer"]

        print(f"A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"⏱  {elapsed:.1f}s | {len(answer.split())} words")

        # Run all checks
        score = 0
        checks = 0

        # 1. Accuracy
        accurate, missing_kw = check_accuracy(answer, test)
        checks += 1
        if accurate:
            score += 1
            print(f"  Accuracy: all expected keywords found")
        else:
            print(f"  Accuracy: missing keywords: {missing_kw}")

        # 2. No hallucination
        no_hallucination, hallucinated = check_no_hallucination(answer, test)
        checks += 1
        if no_hallucination:
            score += 1
            print(f"  No hallucination")
        else:
            print(f"  Hallucination: found forbidden phrases: {hallucinated}")

        # 3. Conciseness (word count)
        concise, word_count = check_conciseness(answer, test)
        checks += 1
        if concise:
            score += 1
            print(f"  Concise: {word_count}/{test.max_words} words")
        else:
            print(f"  Too verbose: {word_count}/{test.max_words} words")

        # 4. No filler
        no_filler, filler_found = check_no_filler(answer)
        checks += 1
        if no_filler:
            score += 1
            print(f"  No filler phrases")
        else:
            print(f"  Filler detected: {filler_found}")

        # 5. Citation (if applicable)
        if test.should_cite and not test.should_refuse:
            cited = check_citation(answer, test)
            checks += 1
            if cited:
                score += 1
                print(f"  Cites source document")
            else:
                print(f"  No source citation found")

        # 6. Refusal (if applicable)
        if test.should_refuse:
            refused = check_refusal(answer, test)
            checks += 1
            if refused:
                score += 1
                print(f"  Correctly refused to answer")
            else:
                print(f"  Should have refused but didn't")

        test_pct = (score / checks * 100) if checks > 0 else 0
        print(f"  Score: {score}/{checks} ({test_pct:.0f}%)")
        print()

        total_score += score
        max_score += checks

        results.append({
            "name": test.name,
            "question": test.question,
            "answer": answer,
            "score": score,
            "max_score": checks,
            "word_count": word_count,
            "response_time": round(elapsed, 2),
            "accurate": accurate,
            "concise": concise,
            "no_filler": no_filler,
        })

        # Clear memory between tests for isolation
        retriever.clear_memory()

    #  Summary 
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    overall_pct = (total_score / max_score * 100) if max_score > 0 else 0
    print(f"Overall Score: {total_score}/{max_score} ({overall_pct:.1f}%)")
    print()

    # Per-dimension scores
    accuracy_pass = sum(1 for r in results if r["accurate"])
    concise_pass = sum(1 for r in results if r["concise"])
    filler_pass = sum(1 for r in results if r["no_filler"])
    avg_words = sum(r["word_count"] for r in results) / len(results)
    avg_time = sum(r["response_time"] for r in results) / len(results)

    print(f"Accuracy:    {accuracy_pass}/{len(results)} tests passed")
    print(f"Conciseness: {concise_pass}/{len(results)} tests passed")
    print(f"No filler:   {filler_pass}/{len(results)} tests passed")
    print(f"Avg words:   {avg_words:.0f}")
    print(f"Avg time:    {avg_time:.1f}s")
    print()

    # Grade
    if overall_pct >= 90:
        grade = "A — Production ready"
    elif overall_pct >= 75:
        grade = "B — Good, minor tuning needed"
    elif overall_pct >= 60:
        grade = "C — Acceptable, needs prompt refinement"
    else:
        grade = "D — Needs significant work"
    print(f"Grade: {grade}")
    print("=" * 70)

    # Save detailed results
    with open("eval_results.json", "w") as f:
        json.dump({
            "mode": config.MODE,
            "overall_score": f"{total_score}/{max_score}",
            "overall_pct": round(overall_pct, 1),
            "grade": grade,
            "tests": results,
        }, f, indent=2)
    print("📊 Detailed results saved to eval_results.json")


if __name__ == "__main__":
    run_evaluation()
