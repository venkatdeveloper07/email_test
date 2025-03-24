from sentence_transformers import SentenceTransformer, util
import re
from pymongo import MongoClient


class LoanEmailProcessor:
    """Processes loan emails, analyzes them, saves to MongoDB, and retrieves/processes."""

    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="loan_emails", collection_name="emails", model_name='all-mpnet-base-v2'):
        """Initializes the processor with MongoDB connection, Sentence Transformer model, and loan categories."""
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.email_collection = self.db[collection_name]
        self.model = SentenceTransformer(model_name)
        self.loan_model_collection = self.db["loan_models"]

        self.loan_request_model = self._load_loan_model()
        self.category_embeddings = self._embed_categories()

    def _load_loan_model(self):
        """Loads the loan request model from MongoDB."""
        model_doc = self.loan_model_collection.find_one(
            {"model_id": "loan_model"})  # assuming there is a doc with model_id "loan_model"
        if model_doc:
            return model_doc["model"]
        else:
            # Default model if not found in MongoDB
            default_model = {
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
        return default_model

    def _embed_categories(self):
        """Pre-computes embeddings for category and subtype descriptions."""
        embeddings = {}
        for request_type, subtypes in self.loan_request_model.items():
            for subtype, data in subtypes.items():
                embeddings[(request_type, subtype)] = self.model.encode(
                    data["description"])
        return embeddings

    def _extract_fields(self, text):
        """Extracts relevant fields (amounts, account numbers, dates) from text."""
        fields = {}
        amount_matches = re.findall(
            r'[$£¥]\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
        if amount_matches:
            fields["loan_amounts"] = amount_matches
        account_matches = re.findall(r'\b\d{6,20}\b', text)
        if account_matches:
            fields["account_numbers"] = account_matches
        date_matches = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', text)
        if date_matches:
            fields["dates"] = date_matches
        return fields

    def analyze_email(self, subject, body, attachments=None):
        """Analyzes a loan email to determine request type, subtype, intent, 
           extracts fields, and provides a confidence score."""

        email_text = subject + " " + body
        if attachments:
            email_text += " " + " ".join(attachments)

        email_embedding = self.model.encode(email_text)

        similarities = {}
        for (request_type, subtype), embedding in self.category_embeddings.items():
            similarities[(request_type, subtype)] = util.cos_sim(
                email_embedding, embedding).item()

        best_match = max(similarities, key=similarities.get)
        confidence = similarities[best_match]
        request_type, subtype = best_match

        extracted_fields = self._extract_fields(email_text)
        intent = self.loan_request_model[request_type][subtype]["description"]

        return {
            "request_type": request_type,
            "subtype": subtype,
            "confidence": confidence,
            "intent": intent,
            "extracted_fields": extracted_fields,
        }

    def get_first_n_emails_with_analysis(self, n=10):
        """Retrieves and analyzes the first n emails from MongoDB."""
        emails = []
        cursor = self.email_collection.find().limit(n)  # Get the first n records

        for email in cursor:
            subject = email.get("subject")
            body = email.get("body")
            attachments = email.get("attachments")

            # Process attachments to get the text for analysis
            attachment_text = ""
            if attachments:
                for attachment in attachments:
                    attachment_type = attachment.get("type")
                    attachment_content = attachment.get("content")
                    if attachment_type == "msg":
                        try:
                            msg = attachment
                            attachment_text += msg["subject"] + \
                                " " + msg["body"] + " "
                        except Exception as e:
                            print(f"Error processing MSG attachment: {e}")
                    elif attachment_content:
                        attachment_text += attachment_content + " "

            # Analyze the email content (including attachment text)
            analysis_results = self.analyze_email(
                subject, body, attachment_text)

            emails.append({
                "filename": email.get("filename"),
                "subject": subject,
                "body": body,
                "attachments": attachments,
                "analysis": analysis_results,
            })
        return emails


# Example Usage:
processor = LoanEmailProcessor()

# Get and analyze the first 10 emails
analyzed_emails = processor.get_first_n_emails_with_analysis(n=10)

# Print the analyzed emails
for email in analyzed_emails:
    print("--- Email ---")
    print("File name:", email["filename"])
    print("Subject:", email["subject"])
    print("Body:", email["body"])
    print("Analysis:", email["analysis"])
    print("---")
