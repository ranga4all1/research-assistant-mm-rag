from datetime import datetime
from typing import Dict, List, Any
import os
import json

import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues, TestColumnDrift
from evidently.ui.workspace import Workspace

class RAGMonitor:
    def __init__(self, workspace_path: str = "./monitoring"):
        """Initialize the RAG monitoring system."""
        self.workspace_path = workspace_path
        self.metrics_log_path = os.path.join(workspace_path, "metrics_log.json")
        self._setup_workspace()

    def _setup_workspace(self) -> None:
        """Setup workspace and files for monitoring."""
        if not os.path.exists(self.workspace_path):
            os.makedirs(self.workspace_path)

        # Initialize metrics log file if it doesn't exist
        if not os.path.exists(self.metrics_log_path):
            with open(self.metrics_log_path, 'w') as f:
                json.dump([], f)

    def log_query(self,
                  query: str,
                  retrieved_docs: pd.DataFrame,
                  answer: str,
                  feedback: Dict[str, Any] = None) -> None:
        """Log a single RAG interaction."""
        timestamp = datetime.now().isoformat()

        # Calculate retrieval diversity (unique document types)
        retrieval_diversity = len(retrieved_docs['type'].unique())

        # Prepare metrics
        metrics = {
            'timestamp': timestamp,
            'query': query,
            'query_length': len(query),
            'num_retrieved_docs': len(retrieved_docs),
            'avg_retrieval_score': float(retrieved_docs['score'].mean()),
            'max_retrieval_score': float(retrieved_docs['score'].max()),
            'min_retrieval_score': float(retrieved_docs['score'].min()),
            'score_std': float(retrieved_docs['score'].std()),
            'retrieval_diversity': retrieval_diversity,
            'answer_length': len(answer),
        }

        # Add feedback metrics if available
        if feedback:
            metrics.update(feedback)

        # Load existing metrics
        with open(self.metrics_log_path, 'r') as f:
            metrics_log = json.load(f)

        # Append new metrics
        metrics_log.append(metrics)

        # Save updated metrics
        with open(self.metrics_log_path, 'w') as f:
            json.dump(metrics_log, f)


    def create_performance_report(self) -> Report:
        """Create a report analyzing RAG system performance."""
        # Load metrics data
        with open(self.metrics_log_path, 'r') as f:
            metrics_log = json.load(f)

        if not metrics_log:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(metrics_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Split data into reference and current
        split_point = len(df) // 2
        reference_data = df.iloc[:split_point]
        current_data = df.iloc[split_point:]

        # Filter out empty columns in reference and current data
        non_empty_columns = reference_data.dropna(axis=1, how='all').columns
        reference_data = reference_data[non_empty_columns]
        current_data = current_data[non_empty_columns]

        # Set up column mapping based on available columns
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = [
            col for col in [
                "query_length", "num_retrieved_docs", "avg_retrieval_score",
                "max_retrieval_score", "min_retrieval_score", "score_std",
                "retrieval_diversity", "answer_length"
            ] if col in non_empty_columns
        ]

        # Create report if data is sufficient
        if not reference_data.empty and not current_data.empty:
            report = Report(metrics=[
                DatasetDriftMetric(),
                DatasetMissingValuesMetric(),
                ColumnDriftMetric(column_name="avg_retrieval_score"),
                ColumnDriftMetric(column_name="num_retrieved_docs")
            ])

            report.run(reference_data=reference_data,
                       current_data=current_data,
                       column_mapping=column_mapping)

            # Save report
            report.save_html(os.path.join(self.workspace_path, "performance_report.html"))

            return report
        return None


    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of monitored metrics."""
        with open(self.metrics_log_path, 'r') as f:
            metrics_log = json.load(f)

        if not metrics_log:
            return {
                'total_queries': 0,
                'message': 'No queries logged yet'
            }

        df = pd.DataFrame(metrics_log)

        # Calculate time-based metrics
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent_df = df[df['timestamp'] > pd.Timestamp.now() - pd.Timedelta(hours=24)]

        return {
            'total_queries': len(df),
            'queries_last_24h': len(recent_df),
            'avg_retrieval_score': round(df['avg_retrieval_score'].mean(), 3),
            'avg_num_docs': round(df['num_retrieved_docs'].mean(), 1),
            'avg_answer_length': round(df['answer_length'].mean(), 1),
            'retrieval_quality': {
                'min_score': round(df['min_retrieval_score'].mean(), 3),
                'max_score': round(df['max_retrieval_score'].mean(), 3),
                'score_std': round(df['score_std'].mean(), 3)
            },
            'diversity': {
                'avg_diversity': round(df['retrieval_diversity'].mean(), 2)
            }
        }

class RAGFeedbackCollector:
    """Collect user feedback for RAG system monitoring."""

    @staticmethod
    def collect_feedback(answer: str) -> Dict[str, Any]:
        """Collect basic feedback metrics about the RAG response. TODO for next iteration of project."""
        return {
            'answer_relevance': float(np.random.uniform(0, 1)),
            'answer_completeness': float(np.random.uniform(0, 1)),
            'retrieval_precision': float(np.random.uniform(0, 1))
        }

def setup_rag_monitoring(rag_system):
    """Setup monitoring for the RAG system."""
    monitor = RAGMonitor()
    feedback_collector = RAGFeedbackCollector()

    # Modify the RAG system's query method to include monitoring
    original_query = rag_system.query

    def monitored_query(question, top_k=5, model='meta-llama/Llama-Vision-Free'):
        # Get response from original query method
        response = original_query(question, top_k=top_k, model=model)

        # Collect feedback
        feedback = feedback_collector.collect_feedback(response['answer'])

        # Log the interaction
        monitor.log_query(
            query=question,
            retrieved_docs=response['search_results'],
            answer=response['answer'],
            feedback=feedback
        )

        # Generate reports periodically
        monitor.create_performance_report()

        return response

    # Replace the original query method with the monitored version
    rag_system.query = monitored_query

    return monitor
