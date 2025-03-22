from sentence_transformers import SentenceTransformer, util
import numpy as np
import re


loan_request_model = {
    "Loan Application/Origination Requests": {
        "New Loan Application": {
            "keywords": ["new loan", "apply for loan", "loan application", "mortgage application", "personal loan application", "auto loan application"],
            "description": "Requests related to applying for a new loan."
        },
        "Pre-Approval Request": {
            "keywords": ["pre-approval", "preapproval", "loan pre-qualification", "pre-qualified loan"],
            "description": "Requests for pre-approval of a loan amount."
        },
        "Loan Application Status Inquiry": {
            "keywords": ["loan status", "application status", "check loan status", "track application"],
            "description": "Inquiries about the status of a loan application."
        },
        "Document Submission": {
            "keywords": ["submit documents", "upload documents", "loan documents", "provide paperwork"],
            "description": "Requests to submit required loan documentation."
        },
        "Loan application modification": {
            "keywords": ["modify loan application", "change loan application", "edit loan application"],
            "description": "Request to change or edit an existing loan application"
        }
    },
    "Loan Servicing Requests": {
        "Payment Inquiry": {
            "keywords": ["payment inquiry", "loan payment", "due date", "balance", "loan balance"],
            "description": "Questions about loan payments, due dates, or balances."
        },
        "Payment Arrangement": {
            "keywords": ["payment arrangement", "payment schedule", "payment change", "modify payment"],
            "description": "Requests to change payment schedules or amounts."
        },
        "Payoff Request": {
            "keywords": ["payoff request", "loan payoff", "pay off loan", "total payoff amount"],
            "description": "Requests for the total amount to pay off a loan."
        },
        "Loan Statement Request": {
            "keywords": ["loan statement", "loan history", "statement request", "transaction history"],
            "description": "Requests for loan statements or transaction history."
        },
        "Escrow Inquiry": {
            "keywords": ["escrow inquiry", "escrow account", "escrow payment", "escrow balance"],
            "description": "Questions about escrow accounts."
        },
        "Hardship requests": {
            "keywords": ["forbearance", "deferment", "hardship plan", "payment relief"],
            "description": "Requests for temporary payment relief due to financial hardship."
        }
    },
    "Account Management Requests": {
        "Address Change": {
            "keywords": ["address change", "update address", "change mailing address"],
            "description": "Requests to update the address on file."
        },
        "Contact Information Update": {
            "keywords": ["contact update", "phone number change", "email change", "update contact"],
            "description": "Requests to update contact information (phone, email)."
        },
        "Account Access Issues": {
            "keywords": ["account access", "login problem", "password issue", "online account"],
            "description": "Problems accessing online loan accounts."
        },
        "Account Closure": {
            "keywords": ["close account", "account closure", "terminate loan"],
            "description": "Requests to close a loan account after payoff."
        },
        "Document request": {
            "keywords": ["loan documents", "copy of documents", "request documents"],
            "description": "Request for copies of loan documents."
        }
    },
    "Information Requests": {
        "Loan Product Information": {
            "keywords": ["loan product", "loan types", "loan terms", "loan information"],
            "description": "Inquiries about different loan types and terms."
        },
        "Interest Rate Inquiry": {
            "keywords": ["interest rate", "current rates", "loan rates", "interest rate inquiry"],
            "description": "Questions about current interest rates."
        },
        "Eligibility Requirements": {
            "keywords": ["eligibility requirements", "loan criteria", "qualify for loan", "loan eligibility"],
            "description": "Inquiries about loan eligibility criteria."
        },
        "Fee Inquiries": {
            "keywords": ["loan fees", "fees and charges", "loan costs", "fee inquiry"],
            "description": "Questions about loan fees and charges."
        },
        "General loan policy questions": {
            "keywords": ["loan policy", "general loan questions", "loan guidelines"],
            "description": "General policy questions about loan services."
        }
    },
    "Support/Problem Resolution Requests": {
        "Payment Processing Issues": {
            "keywords": ["payment problem", "payment issue", "payment not processed", "payment error"],
            "description": "Problems with loan payments not being processed."
        },
        "Error Correction": {
            "keywords": ["error correction", "correct information", "fix error", "wrong information"],
            "description": "Requests to correct errors in account or loan details."
        },
        "Complaint": {
            "keywords": ["complaint", "file complaint", "customer complaint", "service complaint"],
            "description": "Filing a complaint about loan services."
        },
        "Fraud reporting": {
            "keywords": ["fraud", "report fraud", "suspicious activity", "unauthorized activity"],
            "description": "Reporting suspicious or unauthorized activity."
        }
    },
    "Loan Modification Requests": {
        "Rate Modification": {
            "keywords": ["rate modification", "interest rate change", "lower rate", "modify rate"],
            "description": "Requests to change the interest rate."
        },
        "Term Modification": {
            "keywords": ["term modification", "loan term change", "extend loan term", "modify term"],
            "description": "Requests to change the loan term."
        },
        "Principal Reduction": {
            "keywords": ["principal reduction", "reduce principal", "lower principal", "principal balance reduction"],
            "description": "Requests to reduce the loan principal."
        },
        "Forbearance requests": {
            "keywords": ["forbearance", "payment pause", "defer payments"],
            "description": "Requests for a temporary pause in loan payments."
        }
    }
}

def classify_loan_email_semantic(subject, body):
    """Classifies loan emails using semantic similarity and extracts intent."""

    model = SentenceTransformer('all-mpnet-base-v2')

    email_text = subject + " " + body
    email_embedding = model.encode(email_text)

    best_match_category = None
    best_match_subcategory = None
    best_match_score = -1

    for category, subcategories in loan_request_model.items():
        for subcategory, data in subcategories.items():
            description_embedding = model.encode(data["description"])
            similarity = util.cos_sim(email_embedding, description_embedding).item()

            if similarity > best_match_score:
                best_match_score = similarity
                best_match_category = category
                best_match_subcategory = subcategory

    extracted_intent = ""

    if best_match_subcategory:
        extracted_intent = loan_request_model[best_match_category][best_match_subcategory]["description"]

    extracted_fields = {}

    # Example: Extracting potential loan amounts (you can expand this)
    loan_amount_matches = re.findall(r'\$\d+(?:,\d+)?(?:\.\d+)?', email_text)
    if loan_amount_matches:
        extracted_fields["loan_amounts"] = loan_amount_matches

    # Example: Extracting potential account numbers
    account_number_matches = re.findall(r'\b\d{8,16}\b', email_text)  # Adjust regex as needed
    if account_number_matches:
        extracted_fields["account_numbers"] = account_number_matches

    # Example: Extracting dates
    date_matches = re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', email_text)
    if date_matches:
        extracted_fields["dates"] = date_matches

    return {
        "category": best_match_category,
        "subcategory": best_match_subcategory,
        "confidence": best_match_score,
        "extracted_intent": extracted_intent,
        "extracted_fields": extracted_fields,
    }

# Example usage:
subject1 = "New Mortgage Application Inquiry"
body1 = "I want to get a home loan. How do I apply and what are the rates?"

# Example usage:
subject2 = "New Mortgage Application Inquiry"
sender2 = "customer@example.com"
body2 = "I am interested in applying for a new mortgage loan. Could you please provide information about interest rates?"

subject3 = "Account Login Issues"
sender3 = "user@example.com"
body3 = "I am unable to log in to my online loan account. I have tried resetting my password, but it is not working."

subject4 = "Address Change Request"
sender4 = "user2@example.com"
body4 = "I need to update my mailing address for my loan account."

# Example usage:
subject5 = "New Mortgage Application Inquiry"
body5 = "I want to get a home loan of $300,000. My account number is 1234567890. I need to know the terms by 12/25/2024"

print(classify_loan_email_semantic(subject1, body1))
print(classify_loan_email_semantic(subject2, body2))
print(classify_loan_email_semantic(subject3, body3))
print(classify_loan_email_semantic(subject4, body4))
print(classify_loan_email_semantic(subject5, body5))