using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetTrainingDurableFunctions
{
    public class PredictionResult
    {
        public string FeatureLabelTarget { get; set; }

        public double Probability { get; set; }

        public string ConfusionMatrixPredictionClass { get; set; }
    }
}
