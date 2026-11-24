from __future__ import annotations

from model_pipeline import load_and_prepare_dataset, train_and_save_model
from mongodb_service import MongoDBService


def initialize_database_and_model() -> None:
    model_metrics, _, _ = train_and_save_model()
    print("Model trained and saved successfully.")
    print(model_metrics)

    service = MongoDBService()
    if not service.connected:
        print("MongoDB is not running. Start MongoDB before running database scripts.")
        return

    service.ensure_database()

    if service.messages.count_documents({}) == 0:
        dataset = load_and_prepare_dataset()
        for _, row in dataset.head(10).iterrows():
            service.insert_message(row["message"], row["label"], source="dataset")
        print("Seeded message documents with dataset samples.")

    validation_demo = service.invalid_document_demo()
    print("Validation demo:", validation_demo)

    crud_demo = service.demo_crud_operations()
    print("CRUD demo:", crud_demo)

    print("Label distribution:", service.label_distribution())
    print("Label percentages:", service.label_percentages())
    print("Prediction distribution:", service.prediction_distribution())
    print("Average spam probability:", service.average_spam_probability())
    print("Daily prediction stats:", service.daily_prediction_stats())
    print("Feedback statistics:", service.feedback_statistics())


if __name__ == "__main__":
    initialize_database_and_model()
