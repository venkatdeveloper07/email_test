from sentence_transformers import SentenceTransformer, util
import re
import numpy as np

class LoanEmailClassifier:
    """Classifies loan emails based on semantic similarity and extracts relevant information."""

    def __init__(self, model_name='all-mpnet-base-v2'):
        """Initializes the classifier with a Sentence Transformer model and loan categories."""
        self.model = SentenceTransformer(model_name)
        self.loan_request_model = {
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
        self.category_embeddings = self._embed_categories()

    def _embed_categories(self):
        """Pre-computes embeddings for category descriptions."""
        embeddings = {}
        for category, subcategories in self.loan_request_model.items():
            for subcategory, data in subcategories.items():
                embeddings[(category, subcategory)] = self.model.encode(data["description"])
        return embeddings

    def classify(self, subject, body, attachments=None):
        """Classifies a loan email and extracts relevant information."""

        email_text = subject + " " + body
        if attachments:
            email_text += " " + " ".join(attachments)

        email_embedding = self.model.encode(email_text)

        similarities = {}
        for (category, subcategory), embedding in self.category_embeddings.items():
            similarities[(category, subcategory)] = util.cos_sim(email_embedding, embedding).item()

        best_match = max(similarities, key=similarities.get)
        confidence = similarities[best_match]
        category, subcategory = best_match

        extracted_fields = self._extract_fields(subject + " " + body, attachments)

        return {
            "category": category,
            "subcategory": subcategory,
            "confidence": confidence,
            "extracted_intent": self.loan_request_model[category][subcategory]["description"],
            "extracted_fields": extracted_fields,
        }

    def _extract_fields(self, text, attachments=None):
        """Extracts relevant fields (amounts, account numbers, dates) from text and attachments."""
        fields = {}
        full_text = text
        if attachments:
            full_text += " " + " ".join(attachments)

        amount_matches = re.findall(r'\$\d+(?:,\d+)?(?:\.\d+)?', full_text)
        if amount_matches:
            fields["loan_amounts"] = amount_matches
        account_matches = re.findall(r'\b\d{8,16}\b', full_text)
        if account_matches:
            fields["account_numbers"] = account_matches
        date_matches = re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', full_text)
        if date_matches:
            fields["dates"] = date_matches
        return fields

# Example Usage:
classifier = LoanEmailClassifier()

email1 = {
    "subject": "New Mortgage Application Inquiry",
    "body": "I want to get a home loan. How do I apply and what are the rates?",
    "attachments": ["I have attached my income statements and tax returns."]
}
result1 = classifier.classify(email1["subject"], email1["body"], email1.get("attachments"))
print(result1)

email2 = {
    "subject": "Payment Inquiry",
    "body": "When is my next payment due?",
    "attachments": ["20000 debited", "30000 debited", "1000 declined"]
}
result2 = classifier.classify(email2["subject"], email2["body"], email2.get("attachments"))
print(result2)

email3 = {
    "subject": "Loan application modification",
    "body": "I would like to change the information on my loan application.",
}
result3 = classifier.classify(email3["subject"], email3["body"])
print(result3)

email4 = {
    "subject": "New Mortgage Application Inquiry",
    "body": "I want to get a home loan. How do I apply and what are the rates?",
    "attachments": ["I have attached my income statements and tax returns. My account number is 12345678. I want a loan of $200000 by 12/25/2024"]
}
result4 = classifier.classify(email4["subject"], email4["body"], email4.get("attachments"))
print(result4)

email5 = {
    "subject": "Payment Inquiry",
    "body": "When is my next payment due?",
    "attachments": ["My payment is $500. Due date is 04/01/2024"]
}
result5 = classifier.classify(email5["subject"], email5["body"])
print(result5)

email6 = {
    "subject": "Loan application modification",
    "body": "I would like to change the information on my loan application.",
    "attachments": ["My new account number is 87654321."]
}
result6 = classifier.classify(email6["subject"], email6["body"],email6.get("attachments"))
print(result6)