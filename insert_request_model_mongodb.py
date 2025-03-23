from pymongo import MongoClient

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

def insert_loan_model(mongo_uri="mongodb://localhost:27017/", db_name="loan_emails", collection_name="loan_models"):
    """Inserts the loan_request_model into MongoDB."""
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        model_document = {
            "model_id": "loan_model",  # Unique identifier for the model
            "model": loan_request_model,
        }

        # Insert the document
        result = collection.insert_one(model_document)
        print(f"Loan model inserted with ID: {result.inserted_id}")

    except Exception as e:
        print(f"Error inserting loan model: {e}")
    finally:
        if 'client' in locals() and client:
            client.close()

# Example Usage:
insert_loan_model()