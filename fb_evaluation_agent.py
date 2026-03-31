#!/usr/bin/env python3
"""
F&B Company Evaluation Agent
Researches Food & Beverage manufacturing companies and drafts tailored outreach emails
aligned to AI manufacturing optimization solutions.

Usage:
    python fb_evaluation_agent.py "Coca-Cola" --role "VP Operations"
    python fb_evaluation_agent.py "Kraft Heinz" --role "CTO" --name "Jane Smith"
    python fb_evaluation_agent.py "Tyson Foods" --role "Plant Manager" --output result.json
    python fb_evaluation_agent.py --batch contacts.csv --output results/
"""

import os
import json
import argparse
import csv
from pathlib import Path
from typing import Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Our company's solution context (update as needed)
# ---------------------------------------------------------------------------
COMPANY_CONTEXT = """
We are an Atlanta-based B2B software company offering AI platform solutions to optimize
Food & Beverage manufacturing operations.

Our core solutions:
- AI-powered production scheduling and throughput optimization
- Automated quality control and defect detection (vision AI + SPC)
- Predictive maintenance to reduce unplanned downtime
- Real-time OEE (Overall Equipment Effectiveness) monitoring and analytics
- Demand forecasting and supply chain visibility
- Ingredient and recipe optimization for yield improvement
- FDA/FSMA compliance automation and end-to-end traceability
- Energy and waste reduction analytics
- Labor efficiency and workforce scheduling optimization

Differentiators:
- Purpose-built for food & beverage manufacturing (not generic manufacturing)
- Integrates with existing MES, ERP, and SCADA systems (SAP, Oracle, Rockwell, etc.)
- Average customer results: 15-25% OEE improvement, 20-30% reduction in unplanned downtime,
  8-12% yield increase
- Typical deployment: 90 days to production value
"""

# Role-specific solution emphasis and pain point framing
ROLE_PROFILES = {
    "cto": {
        "priorities": "technology modernization, AI/ML integration, data architecture, digital transformation ROI, IT/OT convergence",
        "pain_points": "legacy system debt, data silos across plants, difficulty integrating OT data into analytics platforms, board pressure to show AI strategy",
        "email_tone": "peer-to-peer technical leader, focus on architecture and integration story",
    },
    "cio": {
        "priorities": "enterprise systems integration, data governance, cybersecurity in manufacturing, cloud migration, vendor consolidation",
        "pain_points": "shadow IT on the plant floor, unreliable data from OT systems, compliance audit exposure, managing sprawling tech stack",
        "email_tone": "peer-to-peer technology executive, focus on governance and integration",
    },
    "ceo": {
        "priorities": "competitive differentiation, EBITDA improvement, ESG/sustainability commitments, investor narrative, M&A integration",
        "pain_points": "margin compression from input cost inflation, labor availability, supply chain fragility, activist investor scrutiny",
        "email_tone": "executive peer, focus on business outcomes and strategic positioning",
    },
    "cfo": {
        "priorities": "ROI on capital investments, cost per unit reduction, working capital optimization, CapEx vs OpEx tradeoffs",
        "pain_points": "difficulty quantifying manufacturing efficiency gains, high cost of quality/recalls, energy cost exposure, justifying automation spend",
        "email_tone": "business case focused, lead with financial metrics and payback period",
    },
    "vp operations": {
        "priorities": "OEE improvement, throughput vs. cost tradeoff, labor productivity, multi-plant standardization",
        "pain_points": "inconsistent plant performance, over-reliance on tribal knowledge, manual reporting overhead, changeover inefficiency",
        "email_tone": "operations peer, specific about line-level outcomes and operational metrics",
    },
    "vp manufacturing": {
        "priorities": "production scheduling optimization, yield improvement, waste/shrink reduction, capacity planning",
        "pain_points": "schedule adherence, raw material variability affecting yields, high scrap rates, SKU proliferation complexity",
        "email_tone": "manufacturing peer, focus on plant floor impact and yield metrics",
    },
    "plant manager": {
        "priorities": "shift performance, real-time visibility, downtime reduction, quality compliance, crew accountability",
        "pain_points": "reacting to problems after the fact, paper-based quality logs, explaining variance to corporate, line imbalances",
        "email_tone": "practical and direct, focus on day-to-day operational relief",
    },
    "quality manager": {
        "priorities": "SPC and statistical quality control, automated inspection, FSMA compliance, recall prevention, traceability",
        "pain_points": "manual inspection bottlenecks, CAPA cycle times, audit readiness, consumer complaint trends, difficulty tracing ingredient lots",
        "email_tone": "quality peer, focus on compliance confidence and defect reduction",
    },
    "supply chain": {
        "priorities": "demand forecasting accuracy, inventory optimization, supplier reliability, lead time reduction, S&OP alignment",
        "pain_points": "demand volatility, excess safety stock costs, production-sales disconnects, raw material shortages disrupting schedules",
        "email_tone": "supply chain peer, focus on forecast accuracy and inventory efficiency",
    },
    "default": {
        "priorities": "operational efficiency, cost reduction, quality improvement, competitive advantage, sustainability",
        "pain_points": "rising input costs, labor constraints, regulatory pressure, supply chain disruption, margin erosion",
        "email_tone": "professional and consultative, balance operational and business outcomes",
    },
}

# ---------------------------------------------------------------------------
# Tool definitions for the agentic loop
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for current information about a company, industry trends, "
            "recent news, financial performance, or strategic initiatives. "
            "Use this multiple times with different queries to build a complete picture."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query. Be precise for best results.",
                }
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Web search implementations
# ---------------------------------------------------------------------------
def search_web(query: str) -> str:
    """Dispatch to configured search provider."""
    provider = os.getenv("SEARCH_PROVIDER", "tavily").lower()
    if provider == "tavily":
        return _search_tavily(query)
    elif provider == "brave":
        return _search_brave(query)
    else:
        return f"[No search provider configured — Claude will use its training knowledge for: {query}]"


def _search_tavily(query: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return f"[TAVILY_API_KEY not set — Claude will use its training knowledge for: {query}]"
    try:
        import requests
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 5, "search_depth": "advanced"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return f"No results found for: {query}"
        parts = []
        for r in results[:5]:
            parts.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Content: {r.get('content', '')[:400]}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as exc:
        return f"[Search error ({exc}) — Claude will use training knowledge for: {query}]"


def _search_brave(query: str) -> str:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return f"[BRAVE_API_KEY not set — Claude will use its training knowledge for: {query}]"
    try:
        import requests
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            params={"q": query, "count": 5},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("web", {}).get("results", [])
        if not results:
            return f"No results found for: {query}"
        parts = []
        for r in results[:5]:
            parts.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', '')}\n"
                f"Description: {r.get('description', '')}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as exc:
        return f"[Search error ({exc}) — Claude will use training knowledge for: {query}]"


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------
def load_template(template_path: Optional[str] = None) -> str:
    """Load email template. Falls back to templates/email_template.txt, then built-in default."""
    if template_path:
        p = Path(template_path)
        if not p.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        return p.read_text()

    default_path = Path(__file__).parent / "templates" / "email_template.txt"
    if default_path.exists():
        return default_path.read_text()

    # Minimal built-in fallback — replace this by populating templates/email_template.txt
    return """Subject: {{SUBJECT_LINE}}

Hi {{RECIPIENT_SALUTATION}},

{{OPENING_LINE}}

{{BODY_PARAGRAPH_1}}

{{BODY_PARAGRAPH_2}}

{{CALL_TO_ACTION}}

{{SIGNATURE}}"""


# ---------------------------------------------------------------------------
# Core agent
# ---------------------------------------------------------------------------
def run_agent(
    company_name: str,
    recipient_role: Optional[str] = None,
    recipient_name: Optional[str] = None,
    template_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Research a F&B company and draft a tailored outreach email.

    Returns:
        {
            "company": str,
            "recipient_role": str | None,
            "recipient_name": str | None,
            "primary_needs": list[str],
            "solution_alignment": list[str],
            "email_draft": str,
            "research_steps": int,
        }
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    template = load_template(template_path)

    # Resolve role profile
    role_key = "default"
    if recipient_role:
        for key in ROLE_PROFILES:
            if key in recipient_role.lower():
                role_key = key
                break
    profile = ROLE_PROFILES[role_key]

    system_prompt = f"""You are a B2B sales intelligence agent specializing in Food & Beverage manufacturing.
You work for an Atlanta-based software company. Here is our solution context:

{COMPANY_CONTEXT}

YOUR TASK:
1. Use web_search to research the target company (2–4 searches covering: company overview,
   recent challenges/news, manufacturing operations, sustainability/ESG initiatives).
2. Identify the top 2–3 operational or strategic pain points most relevant to the recipient's role.
3. Map those pain points directly to our specific solutions.
4. Draft a highly personalized outreach email using the provided template — replacing EVERY
   placeholder with researched, specific content.

RECIPIENT ROLE PROFILE ({recipient_role or 'Not specified'}):
- Strategic priorities: {profile['priorities']}
- Likely pain points: {profile['pain_points']}
- Email tone guidance: {profile['email_tone']}

EMAIL TEMPLATE (fill in all placeholders):
---
{template}
---

DRAFTING RULES:
- Replace every {{{{PLACEHOLDER}}}} with specific, researched content.
- Reference real, specific things about the company (initiatives, acquisitions, products, challenges).
- Do NOT use generic F&B platitudes — be specific to this company.
- Connect our solutions to their actual situation, not just the industry broadly.
- Keep tone professional and consultative — a peer reaching out, not a vendor pitching.
- Subject line should reference something specific to the company or role.
- If recipient name is unknown, open with their role title (e.g., "Hi [Name],") as a merge field placeholder.
- End with a low-friction CTA (15-minute call, not a "demo request").

After the email draft, output a brief JSON block tagged <analysis> with:
{{
  "primary_needs": ["need1", "need2", "need3"],
  "solution_alignment": ["our solution X addresses need Y", ...],
  "industry_segment": "e.g. beverage / dairy / snacks",
  "confidence": "high | medium | low"
}}"""

    salutation = recipient_name if recipient_name else f"[{recipient_role or 'Name'}]"
    user_message = (
        f"Research {company_name} and draft a tailored outreach email.\n\n"
        f"Company: {company_name}\n"
        f"Recipient role: {recipient_role or 'Not specified'}\n"
        f"Recipient name: {recipient_name or 'Not specified — use merge field placeholder'}\n"
        f"Recipient salutation for template: {salutation}\n\n"
        f"Start by searching for current information about {company_name}, then draft the email."
    )

    messages = [{"role": "user", "content": user_message}]
    research_steps = 0
    final_text = ""

    for _ in range(12):  # max agentic iterations
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        if verbose:
            print(f"  [stop_reason={response.stop_reason}]")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if verbose:
                        print(f"  [tool={block.name}] {json.dumps(block.input)[:120]}")
                    result = _dispatch_tool(block.name, block.input)
                    research_steps += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

    # Parse embedded analysis block
    analysis = _extract_analysis(final_text)
    # Strip the <analysis> block from the email draft
    email_draft = _strip_analysis_block(final_text).strip()

    return {
        "company": company_name,
        "recipient_role": recipient_role,
        "recipient_name": recipient_name,
        "primary_needs": analysis.get("primary_needs", []),
        "solution_alignment": analysis.get("solution_alignment", []),
        "industry_segment": analysis.get("industry_segment", ""),
        "confidence": analysis.get("confidence", ""),
        "email_draft": email_draft,
        "research_steps": research_steps,
    }


def _dispatch_tool(name: str, inputs: dict) -> str:
    if name == "web_search":
        return search_web(inputs["query"])
    return f"Unknown tool: {name}"


def _extract_analysis(text: str) -> dict:
    """Pull the JSON inside <analysis>...</analysis> tags if present."""
    import re
    match = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    return {}


def _strip_analysis_block(text: str) -> str:
    import re
    return re.sub(r"\s*<analysis>.*?</analysis>", "", text, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Batch processing (for HubSpot CSV export)
# ---------------------------------------------------------------------------
def run_batch(csv_path: str, output_dir: str, template_path: Optional[str] = None, verbose: bool = False):
    """
    Process a CSV file of contacts. Expected columns (case-insensitive):
        company, role, firstname (or first_name), lastname (or last_name)
    Saves individual JSON result files to output_dir.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize column names
        rows = [{k.lower().strip(): v.strip() for k, v in row.items()} for row in reader]

    print(f"Processing {len(rows)} contacts from {csv_path}")

    for i, row in enumerate(rows, 1):
        company = row.get("company") or row.get("company name", "")
        role = row.get("role") or row.get("jobtitle") or row.get("job title", "")
        first = row.get("firstname") or row.get("first_name") or row.get("first name", "")
        last = row.get("lastname") or row.get("last_name") or row.get("last name", "")
        name = f"{first} {last}".strip() or None

        if not company:
            print(f"  [{i}] Skipping row — no company name")
            continue

        print(f"  [{i}/{len(rows)}] {company}" + (f" | {role}" if role else "") + (f" | {name}" if name else ""))

        try:
            result = run_agent(company, role or None, name, template_path, verbose)
            slug = company.lower().replace(" ", "_").replace("/", "-")[:40]
            out_file = output_path / f"{slug}.json"
            out_file.write_text(json.dumps(result, indent=2))
            print(f"           -> Saved: {out_file}")
        except Exception as exc:
            print(f"           -> ERROR: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="F&B Company Evaluation Agent — Draft AI-tailored outreach emails",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fb_evaluation_agent.py "Coca-Cola" --role "VP Operations"
  python fb_evaluation_agent.py "Kraft Heinz" --role "CTO" --name "Jane Smith" --verbose
  python fb_evaluation_agent.py "Tyson Foods" --role "Plant Manager" --output tyson.json
  python fb_evaluation_agent.py --batch contacts.csv --output results/
        """,
    )

    # Single-company mode
    parser.add_argument("company", nargs="?", help="Target company name")
    parser.add_argument("--role", "-r", help="Recipient role/title (e.g., 'VP Operations', 'CTO')")
    parser.add_argument("--name", "-n", help="Recipient's full name")
    parser.add_argument("--template", "-t", help="Path to custom email template file")

    # Batch mode
    parser.add_argument("--batch", help="CSV file path for batch processing")

    # Output options
    parser.add_argument("--output", "-o", help="Save output to file/directory (JSON)")
    parser.add_argument("--email-only", action="store_true", help="Print only the email draft (no metadata)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show agent tool calls as they happen")

    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY environment variable is not set.")

    # Batch mode
    if args.batch:
        if not args.output:
            parser.error("--output <directory> is required for --batch mode.")
        run_batch(args.batch, args.output, args.template, args.verbose)
        return

    # Single-company mode
    if not args.company:
        parser.error("Provide a company name or use --batch for CSV processing.")

    print(f"\nResearching {args.company}..." + (f"  (Role: {args.role})" if args.role else ""))

    result = run_agent(
        company_name=args.company,
        recipient_role=args.role,
        recipient_name=args.name,
        template_path=args.template,
        verbose=args.verbose,
    )

    if args.email_only:
        print(result["email_draft"])
    else:
        print("\n" + "=" * 70)
        print("DRAFTED EMAIL")
        print("=" * 70)
        print(result["email_draft"])
        print("\n" + "=" * 70)
        if result["primary_needs"]:
            print("PRIMARY NEEDS IDENTIFIED:")
            for need in result["primary_needs"]:
                print(f"  • {need}")
        if result["solution_alignment"]:
            print("\nSOLUTION ALIGNMENT:")
            for item in result["solution_alignment"]:
                print(f"  • {item}")
        if result.get("industry_segment"):
            print(f"\nSegment: {result['industry_segment']}  |  Confidence: {result.get('confidence', 'N/A')}")
        print(f"Research steps: {result['research_steps']}")

    if args.output and not args.batch:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nFull output saved to: {out_path}")


if __name__ == "__main__":
    main()
