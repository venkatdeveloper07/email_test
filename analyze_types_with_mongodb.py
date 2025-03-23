from sentence_transformers import SentenceTransformer, util
import re
from pymongo import MongoClient

class LoanEmailProcessor:
    """Processes loan emails, saves them to MongoDB, and retrieves/classifies them."""

    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="loan_emails", collection_name="emails", model_name='all-mpnet-base-v2'):
        """Initializes the processor with MongoDB connection and Sentence Transformer model."""
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
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

    def save_email(self, email_id, subject, body, attachments=None):
        """Saves an email to MongoDB."""
        email_data = {
            "email_id": email_id,
            "subject": subject,
            "body": body,
            "attachments": attachments or [],
        }
        self.collection.insert_one(email_data)

    def get_email(self, email_id):
        """Retrieves an email from MongoDB by email_id."""
        return self.collection.find_one({"email_id": email_id})

    def process_email(self, email_id):
        """Retrieves, classifies, and extracts fields from an email."""
        email = self.get_email(email_id)
        if not email:
            return None

        email_text = email["subject"] + " " + email["body"]
        if email["attachments"]:
            email_text += " " + " ".join(email["attachments"])

        email_embedding = self.model.encode(email_text)

        similarities = {}
        for (category, subcategory), embedding in self.category_embeddings.items():
            similarities[(category, subcategory)] = util.cos_sim(email_embedding, embedding).item()

        best_match = max(similarities, key=similarities.get)
        confidence = similarities[best_match]
        category, subcategory = best_match

        extracted_fields = self._extract_fields(email_text)

        return {
            "category": category,
            "subcategory": subcategory,
            "confidence": confidence,
            "extracted_intent": self.loan_request_model[category][subcategory]["description"],
            "extracted_fields": extracted_fields,
        }

    def _extract_fields(self, text):
        """Extracts relevant fields (amounts, account numbers, dates) from text."""
        fields = {}
        amount_matches = re.findall(r'\$\d+(?:,\d+)?(?:\.\d+)?', text)
        if amount_matches:
            fields["loan_amounts"] = amount_matches
        account_matches = re.findall(r'\b\d{8,16}\b', text)
        if account_matches:
            fields["account_numbers"] = account_matches
        date_matches = re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', text)
        if date_matches:
            fields["dates"] = date_matches
        return fields

# Example Usage:
processor = LoanEmailProcessor()

# Save emails to MongoDB:
processor.save_email("EML0001", "New Mortgage Application Inquiry", "I want to get a home loan.", ["My account number is 12345678. I want a loan of $200000 by 12/25/2024"])
processor.save_email("EML0002", "Payment Inquiry", "When is my next payment due?", ["My payment is $500. Due date is 04/01/2024"])

# Retrieve and process emails:
result1 = processor.process_email("EML0001")
print(result1)

result2 = processor.process_email("EML0002")
print(result2)

result_not_found = processor.process_email("EML0003")
print(result_not_found)