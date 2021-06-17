using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace MLNetTrainingDurableFunctions
{
    public class PerformanceMetrics
    {
        int tps, tns, fps, fns = 0;
        List<string> performanceMatrix = new List<string>(100);

        public PerformanceMetrics(List<string> performanceMatrix, bool calculateBootStrapMetrics = false, int bootStrapIterations = 1000)
        {
            this.performanceMatrix = performanceMatrix;

            // Calculate counts of each metric
            this.tps = performanceMatrix.Count(t => t == "TP");
            this.tns = performanceMatrix.Count(t => t == "TN");
            this.fps = performanceMatrix.Count(t => t == "FP");
            this.fns = performanceMatrix.Count(t => t == "FN");

            if (calculateBootStrapMetrics)
            {
                var seed = 200;
                var random = new Random(seed);

                var mccScores = new List<double>(performanceMatrix.Count);
                var accuracies = new List<double>(performanceMatrix.Count);
                var precisions = new List<double>(performanceMatrix.Count);
                var recalls = new List<double>(performanceMatrix.Count);

                for (int i = 0; i != bootStrapIterations; i++)
                {
                    var bootStapSample = new List<string>(performanceMatrix.Count);

                    for (int j = 0; j != performanceMatrix.Count; j++)
                    {
                        var randomSelectedRecordIndex = random.Next(0, performanceMatrix.Count - 1);
                        var selectedRecord = performanceMatrix[randomSelectedRecordIndex];
                        bootStapSample.Add(selectedRecord);
                    }

                    var mccScore = ((tps * tns - fps * fns) * 1.0) / Math.Sqrt(((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)) * 1.0);
                    mccScores.Add(mccScore);
                    var precision = (tps) * 1.0 / (tps + fps);
                    precisions.Add(precision);
                    var recall = (tps) * 1.0 / (tps + fns);
                    recalls.Add(recall);
                    var accuracy = (tps + tns) * 1.0 / (tps + tns + fps + fns);
                    accuracies.Add(accuracy);
                }

                var mccScoreStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(mccScores.Where(a => !Double.IsNaN(a)));
                var mccScoreAverage = MathNet.Numerics.Statistics.Statistics.Mean(mccScores.Where(a => !Double.IsNaN(a)));
                var precisionstndardDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(precisions.Where(a => !Double.IsNaN(a)));
                var precisionAverage = MathNet.Numerics.Statistics.Statistics.Mean(precisions.Where(a => !Double.IsNaN(a)));
                var accuracyStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(accuracies.Where(a => !Double.IsNaN(a)));
                var accuracyAverage = MathNet.Numerics.Statistics.Statistics.Mean(accuracies.Where(a => !Double.IsNaN(a)));
                var recallStandardDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(recalls.Where(a => !Double.IsNaN(a)));
                var recallAverage = MathNet.Numerics.Statistics.Statistics.Mean(recalls.Where(a => !Double.IsNaN(a)));
            }
        }

        public double Accuracy {
            get
            {
                return (tps + tns) * 1.0 / (tps + tns + fps + fns);
            }
        }

        public double MCCScore
        {
            get
            {
                return ((tps * tns - fps * fns) * 1.0) / Math.Sqrt(((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)) * 1.0);
            }
        }

        public double Precsion
        {
            get
            {
                return (tps) * 1.0 / (tps + fps);
            }
        }

        public double Recall
        {
            get
            {
                return (tps) * 1.0 / (tps + fns);
            }
        }

        public int TruePositives
        {
            get
            {
                return this.tps;
            }
        }

        public int TrueNegatives
        {
            get
            {
                return this.tns;
            }
        }

        public int FalsePositives
        {
            get
            {
                return this.fps;
            }
        }

        public int FalseNegatives
        {
            get
            {
                return this.fns;
            }
        }
    }
}
