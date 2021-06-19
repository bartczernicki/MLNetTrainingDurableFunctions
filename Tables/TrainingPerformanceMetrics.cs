using Microsoft.Azure.Cosmos.Table;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetTrainingDurableFunctions.Tables
{
    public class TrainingPerformanceMetrics : TableEntity
    {
        public TrainingPerformanceMetrics(string TrainingLabel, string modelIdentifier)
        {
            this.PartitionKey = TrainingLabel;
            this.RowKey = modelIdentifier;
        }

        public TrainingPerformanceMetrics() { }

        public string HyperParameters { get; set; }

        public int TruePositives { get; set; }

        public int TrueNegatives { get; set; }

        public int FalsePositives { get; set; }

        public int FalseNegatives { get; set; }

        public double ValidationDataAccuracy { get; set; }

        public double ValidationDataPrecision { get; set; }

        public double ValidationDataRecall { get; set; }

        public double ValidationDataMCCScore { get; set; }

        public double ValidationDataF1Score { get; set; }

        public double BootstrapAccuracyMean { get; set; }

        public double BootstrapPrecisionMean { get; set; }

        public double BootstrapRecallMean { get; set; }

        public double BootstrapMCCScoreMean { get; set; }

        public double BootstrapF1ScoreMean { get; set; }

        public double BootstrapAccuracyStandardDeviation { get; set; }

        public double BootstrapPrecisionStandardDeviation { get; set; }

        public double BootstrapRecallStandardDeviation { get; set; }

        public double BootstrapMCCScoreStandardDeviation { get; set; }

        public double BootstrapF1ScoreStandardDeviation { get; set; }
    }
}
