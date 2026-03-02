"""
test.py — Organizer-side test script
=====================================
This script is run by the hackathon organizers to evaluate a participant's
RAG-based content-moderation system.

It expects:
  1. The participant's Docker container is running and exposes a FastAPI
     server on  http://localhost:8000
  2. The endpoint  POST /analyze  accepts JSON:
         {"prompt": "<user text>"}
     and returns JSON:
         {"harmful": true/false, "articles": ["Article X", ...]}
     - If harmful is false, "articles" should be an empty list [].
     - If harmful is true, "articles" lists violated article numbers.

Usage (organizers):
    docker compose up -d          # start participant container
    python test.py                # run evaluation
    docker compose down           # cleanup
"""

import json
import time
import sys
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{BASE_URL}/analyze"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
TIMEOUT_SECONDS = 120  # max time to wait per request (LLM can be slow)
STARTUP_WAIT = 300     # max seconds to wait for container readiness

# ---------------------------------------------------------------------------
# Test cases: (prompt, expected_harmful, expected_articles_subset)
#   expected_articles_subset: if harmful, at least these articles should appear
# ---------------------------------------------------------------------------
TEST_CASES = [
    # --- Harmful prompts (should detect CCPA violation) ---
    {
        "prompt": "We are selling our customers' personal information to third-party data brokers without informing them or giving them a chance to opt out.",
        "expected_harmful": True,
        "expected_articles_subset": [],  # organizers fill with correct section(s)
        "description": "Selling personal info without opt-out (Sec 1798.120)"
    },
    {
        "prompt": "Our company collects browsing history, geolocation, and biometric data from users but our privacy policy doesn't mention any of this.",
        "expected_harmful": True,
        "expected_articles_subset": [],
        "description": "Undisclosed data collection (Sec 1798.100)"
    },
    {
        "prompt": "A customer asked us to delete their data but we are ignoring their request and keeping all records.",
        "expected_harmful": True,
        "expected_articles_subset": [],
        "description": "Ignoring deletion request (Sec 1798.105)"
    },
    {
        "prompt": "We charge customers who opted out of data selling a higher price for the same service.",
        "expected_harmful": True,
        "expected_articles_subset": [],
        "description": "Discriminatory pricing for opt-out (Sec 1798.125)"
    },
    {
        "prompt": "We are collecting and selling personal data of 14-year-old users without getting their parent's consent.",
        "expected_harmful": True,
        "expected_articles_subset": [],
        "description": "Minor's data without consent (Sec 1798.120)"
    },
    # --- Safe prompts (should NOT detect violation) ---
    {
        "prompt": "Our company provides a clear privacy policy and allows customers to opt out of data selling at any time.",
        "expected_harmful": False,
        "expected_articles_subset": [],
        "description": "CCPA-compliant data practices"
    },
    {
        "prompt": "We deleted all personal data within 45 days after receiving the consumer's verified request.",
        "expected_harmful": False,
        "expected_articles_subset": [],
        "description": "Proper deletion compliance"
    },
    {
        "prompt": "Can we schedule a team meeting for next Monday to discuss the project?",
        "expected_harmful": False,
        "expected_articles_subset": [],
        "description": "Normal meeting request (unrelated to CCPA)"
    },
    {
        "prompt": "Our website has a 'Do Not Sell My Personal Information' link on the homepage as required.",
        "expected_harmful": False,
        "expected_articles_subset": [],
        "description": "Proper opt-out link (compliant)"
    },
    {
        "prompt": "We provide equal service and pricing to all customers regardless of whether they exercise their privacy rights.",
        "expected_harmful": False,
        "expected_articles_subset": [],
        "description": "Non-discriminatory practices (compliant)"
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wait_for_server(url: str, timeout: int = STARTUP_WAIT) -> bool:
    """Block until the participant server is healthy or timeout."""
    print(f"⏳ Waiting for server at {url} (max {timeout}s) ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print("✅ Server is up!\n")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    print("❌ Server did not become ready in time.")
    return False


def validate_response(resp_json: dict) -> list[str]:
    """Return list of validation errors for a single response."""
    errors = []
    if not isinstance(resp_json, dict):
        errors.append(f"Response is not a JSON object: {type(resp_json)}")
        return errors
    if "harmful" not in resp_json:
        errors.append("Missing key 'harmful'")
    elif not isinstance(resp_json["harmful"], bool):
        errors.append(f"'harmful' should be bool, got {type(resp_json['harmful'])}")
    if "articles" not in resp_json:
        errors.append("Missing key 'articles'")
    elif not isinstance(resp_json["articles"], list):
        errors.append(f"'articles' should be list, got {type(resp_json['articles'])}")
    return errors


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_tests() -> dict:
    results = {
        "total": len(TEST_CASES),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "details": []
    }

    for idx, tc in enumerate(TEST_CASES, 1):
        prompt = tc["prompt"]
        desc = tc["description"]
        expected_harmful = tc["expected_harmful"]
        print(f"── Test {idx}/{len(TEST_CASES)}: {desc}")
        print(f"   Prompt : {prompt[:80]}...")

        detail = {"test": idx, "description": desc, "status": "UNKNOWN"}

        try:
            resp = requests.post(
                ANALYZE_ENDPOINT,
                json={"prompt": prompt},
                timeout=TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            resp_json = resp.json()
        except requests.exceptions.Timeout:
            detail["status"] = "ERROR"
            detail["error"] = "Request timed out"
            results["errors"] += 1
            results["details"].append(detail)
            print(f"   ❌ ERROR: Timeout\n")
            continue
        except Exception as e:
            detail["status"] = "ERROR"
            detail["error"] = str(e)
            results["errors"] += 1
            results["details"].append(detail)
            print(f"   ❌ ERROR: {e}\n")
            continue

        # Validate response structure
        v_errors = validate_response(resp_json)
        if v_errors:
            detail["status"] = "FAIL"
            detail["validation_errors"] = v_errors
            results["failed"] += 1
            results["details"].append(detail)
            print(f"   ❌ FAIL: Validation errors: {v_errors}\n")
            continue

        detail["response"] = resp_json

        # Check harmful correctness
        got_harmful = resp_json["harmful"]
        if got_harmful != expected_harmful:
            detail["status"] = "FAIL"
            detail["reason"] = (
                f"Expected harmful={expected_harmful}, got harmful={got_harmful}"
            )
            results["failed"] += 1
            print(f"   ❌ FAIL: {detail['reason']}")
        else:
            # If harmful, check that articles list is non-empty
            if expected_harmful and len(resp_json.get("articles", [])) == 0:
                detail["status"] = "FAIL"
                detail["reason"] = "harmful=true but articles list is empty"
                results["failed"] += 1
                print(f"   ❌ FAIL: {detail['reason']}")
            elif not expected_harmful and len(resp_json.get("articles", [])) > 0:
                detail["status"] = "FAIL"
                detail["reason"] = "harmful=false but articles list is non-empty"
                results["failed"] += 1
                print(f"   ❌ FAIL: {detail['reason']}")
            else:
                detail["status"] = "PASS"
                results["passed"] += 1
                print(f"   ✅ PASS")

        print(f"   Response: {json.dumps(resp_json)}\n")
        results["details"].append(detail)

    return results


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not wait_for_server(HEALTH_ENDPOINT):
        sys.exit(1)

    results = run_tests()

    print("=" * 60)
    print(f"RESULTS: {results['passed']}/{results['total']} passed, "
          f"{results['failed']} failed, {results['errors']} errors")
    print("=" * 60)

    # Dump full results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("📄 Detailed results saved to test_results.json")

    # Exit code: 0 if all passed
    sys.exit(0 if results["failed"] == 0 and results["errors"] == 0 else 1)
