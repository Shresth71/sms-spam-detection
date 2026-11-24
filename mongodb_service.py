from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.errors import PyMongoError


class MongoDBService:
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017",
        database_name: str = "sms_spam_detection",
    ) -> None:
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[MongoClient] = None
        self.db = None
        self.messages = None
        self.predictions = None
        self.feedback = None
        self.connection_error: Optional[str] = None
        self.connect()

    def connect(self) -> bool:
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=3000)
            self.client.admin.command("ping")
            self.db = self.client[self.database_name]
            self.messages = self.db["messages"]
            self.predictions = self.db["predictions"]
            self.feedback = self.db["feedback"]
            self.connection_error = None
            return True
        except Exception as exc:
            self.client = None
            self.db = None
            self.messages = None
            self.predictions = None
            self.feedback = None
            self.connection_error = str(exc)
            return False

    @property
    def connected(self) -> bool:
        return self.db is not None and self.messages is not None and self.predictions is not None and self.feedback is not None

    def ensure_database(self) -> None:
        if not self.connected:
            raise RuntimeError(f"MongoDB is unavailable: {self.connection_error or 'No connection'}")

        message_validator = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["message", "actual_label", "source", "created_at"],
                "properties": {
                    "message": {"bsonType": "string", "minLength": 1},
                    "actual_label": {"enum": ["ham", "spam"]},
                    "source": {"enum": ["dataset", "user", "manual"]},
                    "created_at": {"bsonType": "date"},
                },
            }
        }

        prediction_validator = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["message", "predicted_label", "spam_probability", "ham_probability", "model_name", "created_at"],
                "properties": {
                    "message": {"bsonType": "string", "minLength": 1},
                    "predicted_label": {"enum": ["ham", "spam"]},
                    "spam_probability": {"bsonType": "double", "minimum": 0, "maximum": 1},
                    "ham_probability": {"bsonType": "double", "minimum": 0, "maximum": 1},
                    "model_name": {"bsonType": "string", "minLength": 1},
                    "created_at": {"bsonType": "date"},
                },
            }
        }

        feedback_validator = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["prediction_id", "message", "predicted_label", "correct_label", "feedback", "created_at"],
                "properties": {
                    "prediction_id": {"bsonType": ["objectId", "string"]},
                    "message": {"bsonType": "string", "minLength": 1},
                    "predicted_label": {"enum": ["ham", "spam"]},
                    "correct_label": {"enum": ["ham", "spam"]},
                    "feedback": {"bsonType": "string", "minLength": 1},
                    "created_at": {"bsonType": "date"},
                },
            }
        }

        validators = {
            "messages": message_validator,
            "predictions": prediction_validator,
            "feedback": feedback_validator,
        }

        existing_collections = set(self.db.list_collection_names())
        for name, validator in validators.items():
            collection = getattr(self, name)
            if name not in existing_collections:
                self.db.create_collection(
                    name,
                    validator=validator,
                    validationAction="error",
                    validationLevel="strict",
                )
            else:
                self.db.command(
                    "collMod",
                    name,
                    validator=validator,
                    validationAction="error",
                    validationLevel="strict",
                )

            collection.create_index([("message", 1)])
            collection.create_index([("created_at", -1)])

        self.messages.create_index([("actual_label", 1)])
        self.predictions.create_index([("predicted_label", 1)])
        self.feedback.create_index([("prediction_id", 1)])

    def insert_message(self, message: str, actual_label: str, source: str = "user") -> str:
        cleaned = message.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        if actual_label.lower() not in {"ham", "spam"}:
            raise ValueError("actual_label must be either 'ham' or 'spam'.")

        payload = {
            "message": cleaned,
            "actual_label": actual_label.lower(),
            "source": source,
            "created_at": datetime.now(timezone.utc),
        }
        insert_result = self.messages.insert_one(payload)
        return str(insert_result.inserted_id)

    def insert_prediction(
        self,
        message: str,
        predicted_label: str,
        spam_probability: float,
        ham_probability: float,
        model_name: str = "Multinomial Naive Bayes",
    ) -> str:
        cleaned = message.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty.")
        if predicted_label.lower() not in {"ham", "spam"}:
            raise ValueError("predicted_label must be either 'ham' or 'spam'.")
        if not (0 <= spam_probability <= 1 and 0 <= ham_probability <= 1):
            raise ValueError("Probability values must be between 0 and 1.")

        payload = {
            "message": cleaned,
            "predicted_label": predicted_label.lower(),
            "spam_probability": float(spam_probability),
            "ham_probability": float(ham_probability),
            "model_name": model_name,
            "created_at": datetime.now(timezone.utc),
        }
        result = self.predictions.insert_one(payload)
        return str(result.inserted_id)

    def insert_feedback(
        self,
        prediction_id: str,
        message: str,
        predicted_label: str,
        correct_label: str,
        feedback: str,
    ) -> str:
        if not message.strip():
            raise ValueError("Message cannot be empty.")
        if predicted_label.lower() not in {"ham", "spam"}:
            raise ValueError("predicted_label must be either 'ham' or 'spam'.")
        if correct_label.lower() not in {"ham", "spam"}:
            raise ValueError("correct_label must be either 'ham' or 'spam'.")
        if not feedback.strip():
            raise ValueError("Feedback cannot be empty.")

        try:
            prediction_object_id = ObjectId(prediction_id)
        except InvalidId as exc:
            raise ValueError("Invalid MongoDB ObjectId for prediction_id.") from exc

        payload = {
            "prediction_id": prediction_object_id,
            "message": message.strip(),
            "predicted_label": predicted_label.lower(),
            "correct_label": correct_label.lower(),
            "feedback": feedback.strip(),
            "created_at": datetime.now(timezone.utc),
        }
        result = self.feedback.insert_one(payload)
        return str(result.inserted_id)

    def get_all_messages(self) -> List[Dict[str, Any]]:
        return list(self.messages.find({}, {"_id": 1, "message": 1, "actual_label": 1, "source": 1, "created_at": 1}).sort("created_at", -1))

    def get_messages_by_label(self, label: str) -> List[Dict[str, Any]]:
        label_name = label.lower()
        return list(self.messages.find({"actual_label": label_name}, {"_id": 1, "message": 1, "actual_label": 1, "source": 1, "created_at": 1}).sort("created_at", -1))

    def search_messages(self, keyword: str) -> List[Dict[str, Any]]:
        return list(self.messages.find({"message": {"$regex": keyword, "$options": "i"}}, {"_id": 1, "message": 1, "actual_label": 1, "source": 1, "created_at": 1}).sort("created_at", -1))

    def get_prediction_history(self) -> List[Dict[str, Any]]:
        return list(self.predictions.find({}, {"_id": 1, "message": 1, "predicted_label": 1, "spam_probability": 1, "ham_probability": 1, "model_name": 1, "created_at": 1}).sort("created_at", -1))

    def update_message_label(self, message_id: str, new_label: str) -> bool:
        if new_label.lower() not in {"ham", "spam"}:
            raise ValueError("Label must be either 'ham' or 'spam'.")
        try:
            object_id = ObjectId(message_id)
        except InvalidId as exc:
            raise ValueError("Invalid MongoDB ObjectId for message_id.") from exc

        result = self.messages.update_one({"_id": object_id}, {"$set": {"actual_label": new_label.lower()}})
        return result.modified_count > 0

    def update_feedback(self, feedback_id: str, new_feedback: str) -> bool:
        if not new_feedback.strip():
            raise ValueError("Feedback cannot be empty.")
        try:
            object_id = ObjectId(feedback_id)
        except InvalidId as exc:
            raise ValueError("Invalid MongoDB ObjectId for feedback_id.") from exc

        result = self.feedback.update_one({"_id": object_id}, {"$set": {"feedback": new_feedback.strip()}})
        return result.modified_count > 0

    def delete_message(self, message_id: str) -> bool:
        try:
            object_id = ObjectId(message_id)
        except InvalidId as exc:
            raise ValueError("Invalid MongoDB ObjectId for message_id.") from exc
        result = self.messages.delete_one({"_id": object_id})
        return result.deleted_count > 0

    def delete_prediction(self, prediction_id: str) -> bool:
        try:
            object_id = ObjectId(prediction_id)
        except InvalidId as exc:
            raise ValueError("Invalid MongoDB ObjectId for prediction_id.") from exc
        result = self.predictions.delete_one({"_id": object_id})
        return result.deleted_count > 0

    def label_distribution(self) -> List[Dict[str, Any]]:
        pipeline = [{"$group": {"_id": "$actual_label", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
        return list(self.messages.aggregate(pipeline))

    def label_percentages(self) -> List[Dict[str, Any]]:
        pipeline = [{"$group": {"_id": "$actual_label", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
        records = list(self.messages.aggregate(pipeline))
        total_count = sum(item["count"] for item in records)
        result = []
        for item in records:
            percentage = (item["count"] / total_count * 100) if total_count else 0.0
            result.append({"label": item["_id"], "count": item["count"], "percentage": round(percentage, 2)})
        return result

    def prediction_distribution(self) -> List[Dict[str, Any]]:
        pipeline = [{"$group": {"_id": "$predicted_label", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
        return list(self.predictions.aggregate(pipeline))

    def total_predictions(self) -> int:
        pipeline = [{"$group": {"_id": None, "total_predictions": {"$sum": 1}}}]
        result = list(self.predictions.aggregate(pipeline))
        return int(result[0]["total_predictions"]) if result else 0

    def average_spam_probability(self) -> float:
        pipeline = [{"$group": {"_id": None, "avg_spam_probability": {"$avg": "$spam_probability"}}}]
        result = list(self.predictions.aggregate(pipeline))
        return float(result[0]["avg_spam_probability"]) if result else 0.0

    def repeated_message_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        pipeline = [
            {"$group": {"_id": "$message", "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
            {"$limit": limit},
        ]
        return list(self.predictions.aggregate(pipeline))

    def daily_prediction_stats(self) -> List[Dict[str, Any]]:
        pipeline = [
            {"$project": {"date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}}},
            {"$group": {"_id": "$date", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}},
        ]
        return list(self.predictions.aggregate(pipeline))

    def feedback_statistics(self) -> List[Dict[str, Any]]:
        pipeline = [
            {"$group": {"_id": "$feedback", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        return list(self.feedback.aggregate(pipeline))

    def invalid_document_demo(self) -> Dict[str, Any]:
        if not self.connected:
            return {"status": "not_connected", "message": "MongoDB is not running."}
        try:
            self.messages.insert_one({
                "message": "",
                "actual_label": "spam",
                "source": "dataset",
                "created_at": datetime.now(timezone.utc),
            })
            return {"status": "accepted", "message": "Validation failed to reject empty message."}
        except PyMongoError as exc:
            return {"status": "rejected", "message": str(exc)}

    def demo_crud_operations(self) -> Dict[str, Any]:
        if not self.connected:
            return {"status": "not_connected", "message": "MongoDB is not running."}

        inserted_ids = {}
        inserted_ids["message"] = self.insert_message("You won a free prize!", "spam", source="dataset")
        inserted_ids["prediction"] = self.insert_prediction("You won a free prize!", "spam", 0.96, 0.04)
        inserted_ids["feedback"] = self.insert_feedback(
            inserted_ids["prediction"],
            "You won a free prize!",
            "spam",
            "ham",
            "Incorrect prediction",
        )

        all_messages = self.get_all_messages()
        spam_messages = self.get_messages_by_label("spam")
        keyword_results = self.search_messages("free")
        history = self.get_prediction_history()

        self.update_message_label(inserted_ids["message"], "ham")
        self.update_feedback(inserted_ids["feedback"], "Incorrect prediction - revised")
        self.delete_message(inserted_ids["message"])
        self.delete_prediction(inserted_ids["prediction"])
        return {
            "status": "success",
            "inserted_ids": inserted_ids,
            "counts": {
                "all_messages": len(all_messages),
                "spam_messages": len(spam_messages),
                "keyword_results": len(keyword_results),
                "prediction_history": len(history),
            },
        }
