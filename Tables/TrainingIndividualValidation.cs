using Microsoft.Azure.Cosmos.Table;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetTrainingDurableFunctions.Tables
{
    class TrainingIndividualValidation : TableEntity
    {
        public TrainingIndividualValidation(string TrainingLabel, string baseballBatterIdentifier)
        {
            this.PartitionKey = TrainingLabel;
            this.RowKey = baseballBatterIdentifier;
        }

        public string FullPlayerName { get; set; }

        public bool InductedToHallOfFame { get; set; }

        public bool OnHallOfFameBallot { get; set; }

        public string ConfusionMatrixPredictionClass { get; set; }

        public double PredictionProbability { get; set; }
    }
}
