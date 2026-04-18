import re
import pandas as pd
from task_helper.work.tools.paths import DATABASE_DIR
from task_helper.work.tools.tooling import tool

def _load_email_addresses() -> list[str]:
    addrs: set[str] = set()

    emails = pd.read_csv(DATABASE_DIR / "emails.csv", dtype=str)
    addrs |= set(emails["sender/recipient"].dropna().astype(str).str.lower().tolist())

    calendar = pd.read_csv(DATABASE_DIR / "calendar_events.csv", dtype=str)
    addrs |= set(calendar["participant_email"].dropna().astype(str).str.lower().tolist())

    project_tasks = pd.read_csv(DATABASE_DIR / "project_tasks.csv", dtype=str)
    addrs |= set(project_tasks["assigned_to_email"].dropna().astype(str).str.lower().tolist())

    crm = pd.read_csv(DATABASE_DIR / "customer_relationship_manager_data.csv", dtype=str)
    addrs |= set(crm["assigned_to_email"].dropna().astype(str).str.lower().tolist())
    if "customer_email" in crm.columns:
        addrs |= set(crm["customer_email"].dropna().astype(str).str.lower().tolist())

    # Keep only things that look like emails.
    addrs = {a for a in addrs if "@" in a and "." in a}
    return sorted(addrs)


EMAIL_ADDRESSES = _load_email_addresses()


def _tokenize(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z]+", s.lower()) if t]


def _email_name_tokens(email_address: str) -> set[str]:
    local = email_address.split("@", 1)[0]
    local = local.replace(".", " ").replace("_", " ").replace("-", " ")
    return set(_tokenize(local))


@tool("company_directory.find_email_address", return_direct=False)
def find_email_address(name=""):
    """
    Finds the email address of an employee by their name.

    Parameters
    ----------
    name : str, optional
        Name of the person.

    Returns
    -------
    email_address : str
        Email addresses of the person.

    Examples
    --------
    >>> directory.find_email_address("John")
    "john.smith@example.com"
    """
    if name == "":
        return "Name not provided."
    query_tokens = _tokenize(name)
    if not query_tokens:
        return "Name not provided."

    matches: list[str] = []
    for email_address in EMAIL_ADDRESSES:
        tokens = _email_name_tokens(email_address)
        if all(t in tokens for t in query_tokens):
            matches.append(email_address)

    return matches if matches else "No email address found."
