using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Globalization;
using MathNet.Numerics;

namespace MLNetTrainingDurableFunctions
{
    public static class Resampling
    {
        public static List<string> GenerateBootstrapSample(List<string> empiricalMatrix)
        {
            List<string> bootStrapSample = new List<string>();

            var seed = 200;
            var random = new Random(seed);

            var mccScores = new List<double>(1000);
            var precisions = new List<double>(1000);

            for (int i = 0; i != 500; i++)
            {
                var bootStapSample = new List<string>(empiricalMatrix.Count);

                for (int j = 0; j != empiricalMatrix.Count; j++)
                {
                    var randomSelectedRecordIndex = random.Next(0, empiricalMatrix.Count - 1);
                    var selectedRecord = empiricalMatrix[randomSelectedRecordIndex];
                    bootStapSample.Add(selectedRecord);
                }

                var tps = bootStapSample.Count(t => t == "TP");
                var tns = bootStapSample.Count(t => t == "TN");
                var fps = bootStapSample.Count(t => t == "FP");
                var fns = bootStapSample.Count(t => t == "FN");

                var mccScore = ((tps * tns - fps * fns) * 1.0) / Math.Sqrt(((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)) * 1.0);
                mccScores.Add(mccScore);
                var precision = (tps) * 1.0 / (tps + fps);
                precisions.Add(precision);
            }

            var stndardDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(mccScores.Where(a => !Double.IsNaN(a)));
            var average = MathNet.Numerics.Statistics.Statistics.Mean(mccScores.Where(a => !Double.IsNaN(a)));
            var stndardDevPrecision = MathNet.Numerics.Statistics.Statistics.StandardDeviation(precisions.Where(a => !Double.IsNaN(a)));
            var averagePrecision = MathNet.Numerics.Statistics.Statistics.Mean(precisions.Where(a => !Double.IsNaN(a)));

            return bootStrapSample;
        }
    }
}
