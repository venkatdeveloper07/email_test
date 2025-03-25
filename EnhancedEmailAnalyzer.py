from pymongo import MongoClient
import re
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from textblob import TextBlob
import en_core_web_lg
from typing import Dict, Tuple, List, Optional
import joblib
import os
from datetime import datetime


class EnhancedEmailAnalyzer:
    def __init__(self, mongo_uri="mongodb://localhost:27017/",
                 db_name="loan_emails",
                 collection_name="emails",
                 data_path=None,
                 retrain=False):
        self.nlp = en_core_web_lg.load()
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.email_collection = self.db[collection_name]

        self.model_path = "email_classifier.joblib"
        self._initialize_models()

        if retrain or not os.path.exists(self.model_path):
            if data_path:
                self._train_model(data_path)
            else:
                raise ValueError("No trained model and no data provided")
        else:
            self._load_model()

        # Advanced negation handling
        self.negation_phrases = {
            "not interested", "don't need", "no longer", "cancel my",
            "ignore this", "opt out", "never mind"
        }

        # Define request taxonomy
        self.request_taxonomy = {
            "Financial Services": {
                "keywords": ["loan", "mortgage", "refinance", "interest", "payment"],
                "subcategories": {
                    "Loan": ["Application", "Modification", "Pre-Approval"],
                    "Account": ["Balance", "Statement", "Closure"]
                }
            },
            "Technical Support": {
                "keywords": ["password", "login", "error", "system", "network"],
                "subcategories": {
                    "Access": ["Reset", "Recovery", "Permission"],
                    "Hardware": ["Repair", "Replacement", "Setup"]
                }
            }
        }

    def _initialize_models(self):
        """Initialize ML pipeline"""
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', RandomForestClassifier(
                n_estimators=100, class_weight='balanced'))
        ])

    def _train_model(self, data_path: str) -> None:
        """Train the classification model"""
        try:
            df = pd.read_csv(data_path)
            X = df['text']
            y = df['label']

            if len(X) < 10:
                print("Warning: Small dataset - training without cross-validation")
                self.pipeline.fit(X, y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                self.pipeline.fit(X_train, y_train)
                y_pred = self.pipeline.predict(X_test)
                print(classification_report(y_test, y_pred))

            joblib.dump(self.pipeline, self.model_path)
            print(f"Model saved to {self.model_path}")

        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

    def _load_model(self) -> None:
        """Load pre-trained model"""
        try:
            self.pipeline = joblib.load(self.model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            raise

    def _process_email_with_attachments(self, email: Dict) -> Dict:
        """Process email and all attachments with improved error handling"""
        try:
            # Safely get email fields with defaults
            subject = email.get("subject", "")
            body = email.get("body", "")
            attachments = email.get("attachments", [])

            # Ensure attachments is a list
            if not isinstance(attachments, list):
                attachments = []

            # Combine all content
            combined_text = self._combine_all_content(
                subject, body, attachments)

            # Analyze components
            base_analysis = self._analyze_text(f"{subject}. {body}")

            attachment_analyses = []
            for attachment in attachments:
                try:
                    # Safely access attachment fields
                    if not isinstance(attachment, dict):
                        continue

                    attachment_name = attachment.get("name", "unnamed")
                    attachment_type = attachment.get("type", "unknown")

                    if attachment_type == "msg":
                        text = f"{attachment.get('subject', '')} {attachment.get('body', '')}"
                    else:
                        content = attachment.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(str(item) for item in content)
                        # Convert to string and limit size
                        text = str(content)[:10000]

                    if text.strip():
                        attachment_analyses.append({
                            "name": attachment_name,
                            "type": attachment_type,
                            "analysis": self._analyze_text(text)
                        })
                except Exception as e:
                    print(
                        f"Error processing attachment {attachment_name}: {str(e)}")

            combined_analysis = self._analyze_text(combined_text)

            return {
                "base_analysis": base_analysis,
                "attachments": attachment_analyses,
                "combined_analysis": combined_analysis,
                "extracted_fields": self._extract_fields(combined_text)
            }

        except Exception as e:
            print(f"Error processing email: {str(e)}")
            return {
                "error": str(e),
                "base_analysis": {},
                "attachments": [],
                "combined_analysis": {},
                "extracted_fields": {}
            }

    def analyze_emails(self, limit=10) -> List[Dict]:
        """Analyze emails with comprehensive error handling"""
        results = []
        try:
            for email in self.email_collection.find().limit(limit):
                try:
                    # Ensure email is a dictionary
                    if not isinstance(email, dict):
                        raise ValueError("Email document is not a dictionary")

                    # Process email and attachments
                    analysis = self._process_email_with_attachments(email)
                    results.append({
                        "email_id": str(email.get("_id", "unknown")),
                        "metadata": {
                            "from": email.get("from", "unknown"),
                            "date": email.get("date", "unknown"),
                            "subject": email.get("subject", "")
                        },
                        "analysis": analysis,
                        "processed_at": datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Error processing email: {str(e)}")
                    results.append({
                        "email_id": "error",
                        "error": str(e),
                        "processed_at": datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"MongoDB query failed: {str(e)}")
            results.append({
                "error": f"MongoDB error: {str(e)}",
                "processed_at": datetime.now().isoformat()
            })
        return results

    def _combine_all_content(self, subject: str, body: str, attachments: List[Dict]) -> str:
        """Combine all text content from email and attachments"""
        content_parts = [subject, body]

        for attachment in attachments:
            if attachment.get("type") == "msg":
                content_parts.extend([
                    attachment.get("subject", ""),
                    attachment.get("body", "")
                ])
            else:
                content = attachment.get("content", "")
                if content:
                    content_parts.append(content[:10000])  # Limit size

        return " ".join(content_parts)

    def _analyze_text(self, text: str) -> Dict:
        """Perform complete text analysis"""
        doc = self.nlp(text)

        # Feature extraction
        features = {
            "sentiment": self._get_sentiment(doc),
            "entities": self._extract_entities(doc),
            "urgency": self._detect_urgency(doc),
            "negation": self._detect_negation(doc),
            "tone": self._analyze_tone(doc)
        }

        # Classification
        ml_result = self._ml_predict(text)
        rule_result = self._rule_based_analysis(doc)

        # Use ML result if confidence is high enough, otherwise use rule-based
        final_result = ml_result if ml_result["confidence"] >= 0.6 else rule_result

        return {
            **final_result,
            **features
        }

    def _extract_fields(self, text: str) -> Dict:
        """Extract key fields from text"""
        doc = self.nlp(text)
        return {
            "amounts": [ent.text for ent in doc.ents if ent.label_ == "MONEY"],
            "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
            "people": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
            "organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
            "key_phrases": self._extract_key_phrases(doc)
        }

    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract important noun phrases"""
        return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

    def _ml_predict(self, text: str) -> Dict:
        """Get prediction from ML model"""
        try:
            probas = self.pipeline.predict_proba([text])[0]
            max_idx = np.argmax(probas)
            confidence = probas[max_idx]
            full_label = self.pipeline.classes_[max_idx]

            # Parse hierarchical label
            parts = full_label.split(":")
            category = parts[0]
            subcategory = parts[1] if len(parts) > 1 else "General"
            subtype = parts[2] if len(parts) > 2 else "General"

            return {
                "category": category,
                "subcategory": subcategory,
                "subtype": subtype,
                "confidence": float(confidence),
                "method": "ML"
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {
                "category": "Unknown",
                "subcategory": "General",
                "subtype": "General",
                "confidence": 0.0,
                "method": "Error"
            }

    def _rule_based_analysis(self, doc) -> Dict:
        """Rule-based fallback analysis"""
        text = doc.text.lower()
        matched_category = "Other"
        matched_subcategory = "General"
        matched_subtype = "General"
        confidence = 0.5  # Baseline

        # Category matching
        for category, data in self.request_taxonomy.items():
            if any(kw in text for kw in data["keywords"]):
                matched_category = category
                confidence = 0.7
                break

        # Subcategory matching
        if matched_category != "Other":
            for subcat, subdata in self.request_taxonomy[matched_category]["subcategories"].items():
                # Safely get keywords - check if subdata is a dict first
                if isinstance(subdata, dict):
                    keywords = subdata.get("keywords", [])
                # Handle case where subcategories might be just a list
                elif isinstance(subdata, list):
                    keywords = []
                else:
                    keywords = []

                # Check if any keywords match
                if any(kw.lower() in text for kw in keywords):
                    matched_subcategory = subcat
                    confidence = 0.8
                    break

        # Subtype matching
        if matched_subcategory != "General":
            for subtype in self.request_taxonomy[matched_category]["subcategories"][matched_subcategory]["subtypes"]:
                if subtype.lower() in text:
                    matched_subtype = subtype
                    confidence = 0.9
                    break

        return {
            "category": matched_category,
            "subcategory": matched_subcategory,
            "subtype": matched_subtype,
            "confidence": confidence,
            "method": "Rule-based"
        }

    def _get_sentiment(self, doc) -> float:
        """Get sentiment score using TextBlob (more accurate than VADER for emails)"""
        analysis = TextBlob(doc.text)
        return {
            "polarity": analysis.sentiment.polarity,
            "subjectivity": analysis.sentiment.subjectivity,
            "label": "positive" if analysis.sentiment.polarity > 0 else
            "negative" if analysis.sentiment.polarity < 0 else "neutral"
        }

    def _extract_entities(self, doc) -> List[Dict]:
        """Extract and classify important entities"""
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities

    def _detect_urgency(self, doc) -> Dict:
        """Detect urgency level with context"""
        urgency_terms = {
            "high": ["urgent", "immediately", "asap", "emergency"],
            "medium": ["soon", "prompt", "timely"],
            "low": ["when convenient", "no rush"]
        }

        text = doc.text.lower()
        for level, terms in urgency_terms.items():
            if any(term in text for term in terms):
                return {
                    "level": level,
                    "score": 1.0 if level == "high" else 0.7 if level == "medium" else 0.3,
                    "terms": [t for t in terms if t in text]
                }

        return {"level": "none", "score": 0.0, "terms": []}

    def _detect_negation(self, doc) -> bool:
        """Advanced negation detection with context"""
        # Check for negation phrases
        text = doc.text.lower()
        if any(phrase in text for phrase in self.negation_phrases):
            return True

        # Check dependency-based negation
        for token in doc:
            if token.dep_ == "neg" and token.head.lemma_ in ["want", "need", "require"]:
                return True

        return False

    def _analyze_tone(self, doc) -> str:
        """Analyze email tone based on linguistic features"""
        features = {
            "question_count": sum(1 for sent in doc.sents if "?" in sent.text),
            "imperative_count": sum(1 for token in doc if token.dep_ == "ROOT" and token.tag_ == "VB"),
            "politeness_markers": sum(1 for token in doc if token.lower_ in ["please", "kindly", "appreciate"])
        }

        if features["question_count"] > 2:
            return "inquisitive"
        elif features["imperative_count"] > 3:
            return "demanding" if features["politeness_markers"] < 1 else "assertive"
        elif features["politeness_markers"] > 2:
            return "polite"
        else:
            return "neutral"

    def _handle_negated_request(self, features: Dict) -> Dict:
        """Special handling for negated requests"""
        return {
            "category": "Negated",
            "subcategory": "General",
            "subtype": "General",
            "confidence": 0.95,
            "intent": "Negative request or cancellation",
            "sentiment": features["sentiment"],
            "urgency": features["urgency"],
            "tone": features["tone"],
            "key_entities": features["entities"],
            "method": "Negation detection"
        }


if __name__ == "__main__":
    # Expanded example data with at least 3 examples per class
    example_data = {
        'text': [
            # Financial Services:Loan:Modification (3 examples)
            "I need to modify my loan application details",
            "Please update my mortgage application information",
            "Need to change the terms of my loan",

            # Technical Support:Access:Reset (3 examples)
            "Can't login to my account, need password reset",
            "I forgot my password and need to reset it",
            "Account locked, please reset my credentials",

            # Financial Services:Account:Closure (3 examples)
            "Please cancel my service effective immediately",
            "I want to close my bank account",
            "Terminate my credit card account",

            # Financial Services:Loan:Application (3 examples)
            "When will my mortgage application be processed?",
            "I'd like to apply for a personal loan",
            "Application for home equity loan",

            # Technical Support:System:Error (3 examples)
            "The system keeps giving me error messages",
            "Getting errors when trying to submit forms",
            "Application crashes with error code 500"
        ],
        'label': [
            "Financial Services:Loan:Modification",
            "Financial Services:Loan:Modification",
            "Financial Services:Loan:Modification",

            "Technical Support:Access:Reset",
            "Technical Support:Access:Reset",
            "Technical Support:Access:Reset",

            "Financial Services:Account:Closure",
            "Financial Services:Account:Closure",
            "Financial Services:Account:Closure",

            "Financial Services:Loan:Application",
            "Financial Services:Loan:Application",
            "Financial Services:Loan:Application",

            "Technical Support:System:Error",
            "Technical Support:System:Error",
            "Technical Support:System:Error"
        ]
    }

    # Create temporary training file
    train_file = "training_data.csv"
    pd.DataFrame(example_data).to_csv(train_file, index=False)

    # Initialize analyzer
    analyzer = EnhancedEmailAnalyzer(
        mongo_uri="mongodb://localhost:27017/",
        db_name="loan_emails",
        collection_name="emails",
        data_path="training_data.csv" if os.path.exists(
            "training_data.csv") else None,
        retrain=False
    )

    # Analyze emails
    results = analyzer.analyze_emails(limit=5)

    # Print results
    for result in results:
        print("\n" + "="*80)
        print(f"Email ID: {result['email_id']}")
        print(f"From: {result['metadata']['from']}")
        print(f"Subject: {result['metadata']['subject']}")
        print(f"Date: {result['metadata']['date']}")

        # Combined analysis (main results)
        combined = result['analysis']['combined_analysis']
        print("\nCOMBINED ANALYSIS:")
        print(f"Request Type: {combined['category']}")
        print(f"Sub Type: {combined.get('subcategory', 'N/A')}")
        print(f"Confidence: {combined['confidence']:.2f}")
        print(f"Intent: {combined.get('intent', 'N/A')}")
        print(f"Urgency: {combined['urgency']['level']}")
        print(f"Tone: {combined['tone']}")

        # Extracted fields
        print("\nEXTRACTED FIELDS:")
        fields = result['analysis']['extracted_fields']
        print("Amounts:", ", ".join(fields['amounts'][:3]) or "None")
        print("Dates:", ", ".join(fields['dates'][:3]) or "None")
        print("Key People:", ", ".join(fields['people'][:3]) or "None")
        print("Organizations:", ", ".join(
            fields['organizations'][:3]) or "None")

        # Attachments summary
        if result['analysis']['attachments']:
            print("\nATTACHMENTS:")
            for att in result['analysis']['attachments']:
                print(
                    f"- {att['name']} ({att['type']}): {att['analysis']['category']}")

        print("="*80)
