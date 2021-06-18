using System;
using System.Collections.Generic;
using System.Text;

namespace MLNetTrainingDurableFunctions
{
    public class MLNetTrainingFunctionInput
    {
        public string FeatureLabelTarget { get; set; }

        public MLBBaseballBatter Batter { get; set; }
    }
}
