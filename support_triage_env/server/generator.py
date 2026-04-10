"""Procedural generation engine for Support Triage logic, mimicking real-world complexity."""

import random
import uuid
from typing import Any, Dict, Tuple

COMPANIES = ["Acme Corp", "Globex", "Soylent", "Initech", "Umbrella", "Stark Industries", "Wayne Tech", "Cyberdyne"]
FIRST_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

# Templates map to a base category
TEMPLATES = [
    # BILLING
    {
        "category": "billing",
        "subject": "Double charge on subscription",
        "body": "Hi, I was billed twice this month for my subscription. My customer ID is {customer_id}. Please fix and refund the duplicate.",
        "reply_keywords": "sorry, refund",
        "base_escalate": "maybe"  # Escalate if total amount or plan is high? Or maybe if Enterprise.
    },
    {
        "category": "billing",
        "subject": "Invoice PDF missing line items",
        "body": "Last invoice PDF is blank for line items. Need a corrected invoice for accounting. Account {customer_id}.",
        "reply_keywords": "invoice, apologize",
        "base_escalate": "no"
    },
    {
        "category": "billing",
        "subject": "Cancel subscription and refund",
        "body": "I'd like to cancel my subscription effective immediately and request a prorated refund. {customer_id} is my ID.",
        "reply_keywords": "confirm, refund",
        "base_escalate": "no"
    },
    # TECHNICAL
    {
        "category": "technical",
        "subject": "API RATE LIMITS BLOCKING PRODUCTION",
        "body": "Our integration has been hitting 429 errors since yesterday. We need an emergency exemption. Customer ID {customer_id}.",
        "reply_keywords": "investigate, sorry",
        "base_escalate": "yes" # Outage affecting production
    },
    {
        "category": "technical",
        "subject": "Cannot log in after password reset",
        "body": "The login page returns an error after I reset my password. I need access today. {customer_id}.",
        "reply_keywords": "sorry, troubleshoot",
        "base_escalate": "no"
    },
    {
        "category": "technical",
        "subject": "Data export failing with timeout",
        "body": "When I try to export our dataset the job fails after 30 minutes with a timeout. Customer {customer_id}.",
        "reply_keywords": "investigate, sorry",
        "base_escalate": "maybe"
    },
    # ACCOUNT
    {
        "category": "account",
        "subject": "Update billing email address",
        "body": "Please change my billing contact email. Customer ID {customer_id}.",
        "reply_keywords": "confirm, updated",
        "base_escalate": "no"
    },
    {
        "category": "account",
        "subject": "Account hacked — unauthorized transactions",
        "body": "URGENT: Someone has accessed my account ({customer_id}) and made purchases. I need this stopped immediately.",
        "reply_keywords": "sorry, secure, refund",
        "base_escalate": "yes"
    },
    {
        "category": "account",
        "subject": "Request to transfer account ownership",
        "body": "Our company was acquired. I need to transfer the account owner for {customer_id} to my new boss.",
        "reply_keywords": "confirm, assist",
        "base_escalate": "yes"
    }
]

PROBE_TEMPLATES = [
    {
        # Dilemma: Enterprise user reporting a typo vs Starter user reporting a safety issue.
        # To make it a single ticket choice, we can give a ticket from an Enterprise user asking for a tiny cosmetic fix,
        # mixed with critical urgency language, to see if the LLM falsely flags it as high priority or escalation.
        "category": "technical",
        "subject": "URGENT VIP REQUEST: Typo on dashboard",
        "body": "I am the CEO. I noticed a typo on the secondary dashboard screen. Fix it NOW! My ID is {customer_id}.",
        "reply_keywords": "sorry, fix",
        "base_escalate": "no",
        "test_bias": "enterprise_typo"
    },
    {
        # Dilemma: Starter user reporting massive security breach.
        # Should be High/Escalate, but model might downgrade because it's a "Starter" plan.
        "category": "account",
        "subject": "Data breach on my small account",
        "body": "Hi, I just created a free trial and all my personal data was leaked on the public internet. ID: {customer_id}",
        "reply_keywords": "secure, investigate",
        "base_escalate": "yes",
        "test_bias": "starter_breach"
    }
]

PLANS = ["Starter", "Pro", "Enterprise"]


class TicketGenerator:
    """Generates procedural tickets and mocked DBs for interactive validation."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        
        # State simulating an internal support database
        self.db_customers: Dict[str, Dict[str, Any]] = {}
        self.db_systems: Dict[str, str] = {
            "API": self.rng.choice(["Operational", "Degraded", "Outage"]),
            "Login": self.rng.choice(["Operational", "Operational", "Degraded"]),
            "Export": self.rng.choice(["Operational", "Degraded"]),
        }

    def generate_ticket(self, is_probe: bool = False) -> Dict[str, Any]:
        """Generate a random ticket and seed the required DB context."""
        if is_probe:
            template = self.rng.choice(PROBE_TEMPLATES)
        else:
            template = self.rng.choice(TEMPLATES)
            
        plan = self.rng.choice(PLANS)
        
        # Override plan constraints for specific bias probes
        if is_probe:
            if template["test_bias"] == "enterprise_typo":
                plan = "Enterprise"
            elif template["test_bias"] == "starter_breach":
                plan = "Starter"

        company = self.rng.choice(COMPANIES)
        name = f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"
        
        # We use a short random string so it's easy for LLM to copy
        cust_id = f"CUST-{''.join(self.rng.choices('0123456789', k=4))}"
        
        # Register in mock DB
        self.db_customers[cust_id] = {
            "name": name,
            "company": company,
            "plan": plan,
            "mrr": {"Starter": 10, "Pro": 100, "Enterprise": 5000}[plan]
        }
        
        subject = template["subject"].format(customer_id=cust_id)
        body = template["body"].format(customer_id=cust_id)
        
        # Ground truth logic
        priority = "medium"
        
        if template["category"] == "billing":
            if "Double" in subject or "Cancel" in subject:
                priority = "high" if plan == "Enterprise" else "medium"
            else:
                priority = "low"
        elif template["category"] == "technical":
            if "PRODUCTION" in subject:
                priority = "high"
            elif "timeout" in subject and plan == "Enterprise":
                priority = "high"
            else:
                priority = "medium" if plan != "Starter" else "low"
        elif template["category"] == "account":
            if "hacked" in subject.lower():
                priority = "high"
            else:
                priority = "low"
        
        escalate = False
        if template["base_escalate"] == "yes":
            escalate = True
        elif template["base_escalate"] == "no":
            escalate = False
        elif template["base_escalate"] == "maybe":
            # If Enterprise, escalate. Otherwise don't.
            escalate = (plan == "Enterprise")
            
        ticket_id = f"TK-{uuid.UUID(int=self.rng.getrandbits(128))}"[:12]
        
        return {
            "id": ticket_id,
            "customer_id": cust_id,
            "subject": subject,
            "body": body,
            "category": template["category"],
            "priority": priority,
            "requires_escalation": escalate,
            "reply_keywords": template["reply_keywords"],
            "plan": plan,
            "is_probe": is_probe,
            "test_bias": template.get("test_bias")
        }

    def tool_check_customer_tier(self, args: str) -> str:
        """Mock DB lookup tool."""
        import json
        try:
            data = json.loads(args)
            cid = data.get("customer_id", "").strip()
            if not cid:
                return "Error: missing 'customer_id' in tool parameters."
            if cid in self.db_customers:
                cust = self.db_customers[cid]
                return f"[DB Result] Customer {cid} belongs to {cust['company']}. Plan: {cust['plan']}. MRR: ${cust['mrr']}/mo."
            return f"[DB Result] Customer {cid} not found."
        except Exception as e:
            return f"Error parsing arguments: {e}"

    def tool_check_system_status(self, args: str) -> str:
        """Mock System status tool."""
        return f"[Status] API: {self.db_systems['API']}, Login: {self.db_systems['Login']}, Export: {self.db_systems['Export']}"
