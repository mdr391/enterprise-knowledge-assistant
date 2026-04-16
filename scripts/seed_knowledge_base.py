#!/usr/bin/env python3
"""
Seed script — loads sample enterprise documents into the knowledge base.
Run this after starting the API to have data ready for demo queries.

Usage:
    python scripts/seed_knowledge_base.py
    python scripts/seed_knowledge_base.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

SAMPLE_DOCUMENTS = [
    {
        "title": "Employee Vacation & PTO Policy 2024",
        "tags": ["hr", "policy", "vacation"],
        "content": """
Employee Vacation & PTO Policy — Effective January 2024

VACATION ACCRUAL
Full-time employees accrue vacation at the following rates based on years of service:
- 0–2 years: 15 days per year (1.25 days/month)
- 3–5 years: 20 days per year (1.67 days/month)
- 6+ years: 25 days per year (2.08 days/month)

Part-time employees accrue vacation on a pro-rated basis relative to their scheduled hours.

CARRYOVER
Employees may carry over up to 5 unused vacation days into the following calendar year.
Any days exceeding the carryover limit are forfeited on January 1. Employees are strongly
encouraged to use their vacation within the year it is earned.

REQUESTING TIME OFF
All vacation requests must be submitted through the HR portal at least 10 business days
in advance for requests of 3+ consecutive days, and at least 5 business days in advance
for shorter absences. Manager approval is required before time off is confirmed.

During peak periods (defined by team leads at the start of each quarter), a minimum of
50% of team members must remain available. Requests during peak periods are evaluated
on a first-come, first-served basis.

SICK LEAVE
Employees receive 10 sick days per calendar year, non-accruing and non-transferable.
Sick days do not roll over. For absences exceeding 3 consecutive days, a doctor's note
is required.

BEREAVEMENT LEAVE
Up to 5 paid days for the loss of an immediate family member (spouse, child, parent,
sibling). Up to 3 paid days for extended family (grandparents, in-laws). Additional
unpaid leave may be granted at manager discretion.
        """.strip(),
    },
    {
        "title": "Remote Work & Home Office Expense Policy",
        "tags": ["finance", "remote", "policy"],
        "content": """
Remote Work & Home Office Expense Reimbursement Policy

ELIGIBILITY
All full-time employees approved for remote or hybrid work arrangements are eligible
for home office reimbursement under this policy.

MONTHLY STIPEND
Eligible employees receive a $100/month internet and utilities stipend, paid automatically
on the first payroll of each month. No receipts are required for the monthly stipend.

ONE-TIME EQUIPMENT ALLOWANCE
New remote employees receive a one-time $1,500 equipment allowance for home office setup.
Eligible purchases include: monitors, keyboards, mice, webcams, headsets, desk lamps,
ergonomic chairs, and standing desks. Personal computers and mobile phones are excluded.

Equipment purchased under this allowance is the property of the company if the total
purchase exceeds $500. Items under $500 are considered personal property.

RECURRING EXPENSES
Employees may submit monthly reimbursement claims for up to $50 in additional expenses
such as printer ink, cables, or other supplies directly required for work. Receipts must
be submitted within 30 days of purchase via the Expenses portal.

SUBMISSION PROCESS
1. Log into the internal Expenses portal at expenses.company.internal
2. Select "Remote Work Expense" as the category
3. Upload receipt (JPEG, PNG, or PDF)
4. Submit by the 15th of the following month for same-month reimbursement

Late submissions (after 30 days) will be reviewed on a case-by-case basis and may be
denied. Reimbursement is processed within 2 pay cycles of approval.
        """.strip(),
    },
    {
        "title": "Engineering On-Call Runbook — Incident Response",
        "tags": ["engineering", "oncall", "runbook"],
        "content": """
Engineering On-Call Runbook — Incident Response Procedures

SEVERITY LEVELS
P0 — Critical: Complete service outage or data loss. All hands required.
     Response SLA: Acknowledge within 5 minutes, bridge open within 15 minutes.
P1 — High: Major feature broken, >20% of users affected.
     Response SLA: Acknowledge within 15 minutes, fix or mitigation within 2 hours.
P2 — Medium: Degraded performance or non-critical feature broken.
     Response SLA: Acknowledge within 1 hour, fix within 8 hours.
P3 — Low: Minor bug, cosmetic issue, single-user impact.
     Response SLA: Acknowledge within 4 hours, fix in next sprint.

ESCALATION PATH
1. On-call engineer (primary pager)
2. On-call engineer (secondary pager) — if primary does not acknowledge within SLA
3. Engineering Manager — if secondary does not acknowledge within 10 minutes of escalation
4. VP Engineering — for P0 incidents only, if resolution is not in sight within 30 minutes

INCIDENT BRIDGE
For P0/P1 incidents, open a Zoom bridge immediately:
- Link: zoom.company.internal/incident-bridge
- Announce in #incidents Slack channel with severity, summary, and bridge link
- Designate: Incident Commander (IC), Communications Lead, Technical Lead

POST-INCIDENT REVIEW
All P0 and P1 incidents require a written post-mortem within 3 business days.
Post-mortems must follow the blameless template at: wiki.company.internal/postmortem-template
Key sections: Timeline, Root Cause, Impact, Action Items (with owners and due dates).

MONITORING DASHBOARDS
- Primary: datadog.company.internal/dashboard/prod-overview
- API latency: datadog.company.internal/dashboard/api-latency
- Error rates: datadog.company.internal/dashboard/error-rates
- Infra: datadog.company.internal/dashboard/infrastructure
        """.strip(),
    },
    {
        "title": "AI & Machine Learning Responsible Use Guidelines",
        "tags": ["ai", "policy", "engineering", "governance"],
        "content": """
Responsible AI Use Guidelines — Internal Policy v2.1

PURPOSE
These guidelines govern the development, deployment, and use of AI and machine learning
systems within the company. They apply to all employees, contractors, and vendors
building or using AI-powered tools on behalf of the company.

CORE PRINCIPLES
1. Transparency: AI systems must clearly communicate to users when they are interacting
   with an automated system. Users must never be misled into believing they are
   communicating with a human when they are not.

2. Accountability: Every AI system in production must have a designated owner responsible
   for monitoring its behavior, responding to issues, and periodic review.

3. Fairness: AI systems must be evaluated for bias before deployment. Any model trained
   on historical data must include a bias audit covering protected characteristics.

4. Privacy: AI systems must not process personally identifiable information (PII) without
   an approved data handling agreement. Vendor AI APIs (including LLMs) must be reviewed
   by the Legal & Privacy team before production use.

5. Human Oversight: High-stakes decisions (hiring, performance management, credit,
   medical) must not be fully automated. A human must be in the loop for final decisions.

LLM-SPECIFIC REQUIREMENTS
- All LLM integrations must implement a system prompt that constrains the model's scope.
- Responses must be grounded in verifiable sources where possible (RAG or citations).
- "Jailbreak" resistance testing must be performed before production deployment.
- Output filtering must be applied for any user-facing LLM response.

PROHIBITED USES
- Generating content designed to deceive, manipulate, or defraud.
- Surveillance of employees without explicit consent and legal review.
- Any use case that violates applicable law or company Code of Conduct.

INCIDENT REPORTING
AI-related incidents (unexpected outputs, bias complaints, data leaks) must be reported
to the AI Governance team at ai-governance@company.internal within 24 hours of discovery.
        """.strip(),
    },
    {
        "title": "New Employee Onboarding Checklist",
        "tags": ["hr", "onboarding"],
        "content": """
New Employee Onboarding — 30-Day Checklist

WEEK 1: SETUP & ORIENTATION
Day 1:
- IT will provision your laptop and deliver it to your desk or ship it to your home address.
- Log into Okta (SSO) using credentials emailed to your personal address.
- Complete I-9 verification via the HR portal within 3 business days of start date.
- Join your team's Slack workspace and introduce yourself in #general.
- Schedule 1:1s with your manager and immediate teammates for weeks 1–2.

Days 2–5:
- Complete mandatory training modules in the LMS (Learning Management System):
  * Security Awareness Training (required, ~45 min)
  * Code of Conduct & Ethics (required, ~30 min)
  * Data Privacy Fundamentals (required, ~60 min)
- Set up your development environment using the Engineering Onboarding Guide at
  wiki.company.internal/dev-setup
- Request access to required systems via the Access Request portal.

WEEK 2: RAMP-UP
- Shadow at least 2 team meetings and 1 cross-functional meeting.
- Complete your first pull request (even if trivial) by end of week 2.
- Review the team's current sprint board and understand active priorities.
- Read the last 3 quarterly engineering retrospectives.

WEEKS 3–4: FIRST CONTRIBUTIONS
- Own at least one P3 bug fix or small feature end-to-end.
- Present your changes in a team sync or demo session.
- Complete your 30-day check-in with your manager.
- Submit your equipment allowance claim if working remotely (see Remote Work Policy).

BENEFITS ENROLLMENT
Benefits enrollment must be completed within 30 days of your start date. Late enrollment
is not permitted outside of qualifying life events. Log into benefits.company.internal
to review options and make elections.

KEY CONTACTS
- HR Questions: hr-help@company.internal or Slack #ask-hr
- IT Support: it-support@company.internal or Slack #it-help
- Payroll: payroll@company.internal
        """.strip(),
    },
]


def seed(base_url: str) -> None:
    print(f"Seeding knowledge base at {base_url}...\n")
    success, failed = 0, 0

    for doc in SAMPLE_DOCUMENTS:
        payload = json.dumps(doc).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/ingest/",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                print(f"  ✓  {doc['title']}")
                print(f"     document_id={data['document_id']}  chunks={data['chunks_created']}\n")
                success += 1
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"  ✗  {doc['title']} — HTTP {e.code}: {body}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗  {doc['title']} — {e}\n")
            failed += 1

    print(f"Done. {success} succeeded, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the knowledge base with sample documents")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    seed(args.base_url)
