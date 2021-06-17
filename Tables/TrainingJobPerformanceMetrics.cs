using Microsoft.Azure.Cosmos.Table;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetTrainingDurableFunctions.Tables
{
    public class TrainingJobPerformanceMetrics : TableEntity
    {
        public TrainingJobPerformanceMetrics(string JobName, string Algorithm)
        {
            this.PartitionKey = JobName;
            this.RowKey = Algorithm;
        }

        public TrainingJobPerformanceMetrics() { }

        public string HyperParameters { get; set; }

        public int TruePositives { get; set; }

        public int TrueNegatives { get; set; }

        public int FalsePositives { get; set; }

        public int FalseNegatives { get; set; }

        public double Accuracy { get; set; }

        public double Precision { get; set; }

        public double Recall { get; set; }

        public double MCCScore { get; set; }

        public double AccuracyStandardDeviation { get; set; }

        public double PrecisionStandardDeviation { get; set; }

        public double RecallStandardDeviation { get; set; }

        public double MCCScoreStandardDeviation { get; set; }
    }
}
