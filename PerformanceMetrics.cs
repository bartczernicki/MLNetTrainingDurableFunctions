using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace MLNetTrainingDurableFunctions
{
    public class PerformanceMetrics
    {
        // classification matrix variables
        int tps, tns, fps, fns = 0;
        double mccScoreStandarddDev, accuracyStandarddDev, precisionStandarddDev, recallStandarddDev = 0.0;

        List<string> classificationMatrix = new List<string>(100);

        public PerformanceMetrics(List<string> classificationMatrix, bool calculateBootStrapMetrics = false, int bootStrapIterations = 1000)
        {
            this.classificationMatrix = classificationMatrix;

            // Calculate counts of each metric
            this.tps = classificationMatrix.Count(t => t == "TP");
            this.tns = classificationMatrix.Count(t => t == "TN");
            this.fps = classificationMatrix.Count(t => t == "FP");
            this.fns = classificationMatrix.Count(t => t == "FN");

            if (calculateBootStrapMetrics)
            {
                var seed = 200;
                var random = new Random(seed);

                var mccScores = new List<double>(classificationMatrix.Count);
                var accuracies = new List<double>(classificationMatrix.Count);
                var precisions = new List<double>(classificationMatrix.Count);
                var recalls = new List<double>(classificationMatrix.Count);

                for (int i = 0; i != bootStrapIterations; i++)
                {
                    var bootStapSample = new List<string>(classificationMatrix.Count);

                    for (int j = 0; j != classificationMatrix.Count; j++)
                    {
                        var randomSelectedRecordIndex = random.Next(0, classificationMatrix.Count - 1);
                        var selectedRecord = classificationMatrix[randomSelectedRecordIndex];
                        bootStapSample.Add(selectedRecord);
                    }

                    var tps = bootStapSample.Count(t => t == "TP");
                    var tns = bootStapSample.Count(t => t == "TN");
                    var fps = bootStapSample.Count(t => t == "FP");
                    var fns = bootStapSample.Count(t => t == "FN");

                    var mccScore = PerformanceMetrics.MCCScoreCalculation(tps, tns, fps, fns);
                    mccScores.Add(mccScore);
                    var precision = PerformanceMetrics.PrecisionCalculation(tps, tns, fps, fns);
                    precisions.Add(precision);
                    var recall = PerformanceMetrics.RecallCalculation(tps, tns, fps, fns);
                    recalls.Add(recall);
                    var accuracy = PerformanceMetrics.AccuracyCalculation(tps, tns, fps, fns);
                    accuracies.Add(accuracy);
                }

                this.mccScoreStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(mccScores.Where(a => !Double.IsNaN(a)));
                var mccScoreAverage = MathNet.Numerics.Statistics.Statistics.Mean(mccScores.Where(a => !Double.IsNaN(a)));
                this.precisionStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(precisions.Where(a => !Double.IsNaN(a)));
                var precisionAverage = MathNet.Numerics.Statistics.Statistics.Mean(precisions.Where(a => !Double.IsNaN(a)));
                this.accuracyStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(accuracies.Where(a => !Double.IsNaN(a)));
                var accuracyAverage = MathNet.Numerics.Statistics.Statistics.Mean(accuracies.Where(a => !Double.IsNaN(a)));
                this.recallStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(recalls.Where(a => !Double.IsNaN(a)));
                var recallAverage = MathNet.Numerics.Statistics.Statistics.Mean(recalls.Where(a => !Double.IsNaN(a)));
            }
        }

        public double Accuracy {
            get
            {
                return (tps + tns) * 1.0 / (tps + tns + fps + fns);
            }
        }

        public double AccuracyBootStrapStandardDeviation
        {
            get
            {
                return this.accuracyStandarddDev;
            }
        }

        public double MCCScore
        {
            get
            {
                return ((tps * tns - fps * fns) * 1.0) / Math.Sqrt(((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)) * 1.0);
            }
        }

        public double MCCScoreStandardDeviation
        {
            get
            {
                return this.mccScoreStandarddDev;
            }
        }


        public double Precsion
        {
            get
            {
                return (tps) * 1.0 / (tps + fps);
            }
        }

        public double PrecisionBootStrapStandardDeviation
        {
            get
            {
                return this.precisionStandarddDev;
            }
        }

        public double Recall
        {
            get
            {
                return (tps) * 1.0 / (tps + fns);
            }
        }

        public double RecallBootStrapStandardDeviation
        {
            get
            {
                return this.recallStandarddDev;
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

        public static double AccuracyCalculation(int tps, int tns, int fps, int fns)
        {
            var accuracy = (tps + tns) * 1.0 / (tps + tns + fps + fns);

            return accuracy;
        }

        public static double PrecisionCalculation(int tps, int tns, int fps, int fns)
        {
            var precision = (tps) * 1.0 / (tps + fps);

            return precision;
        }

        public static double RecallCalculation(int tps, int tns, int fps, int fns)
        {
            var recall = (tps) * 1.0 / (tps + fns);

            return recall;
        }

        public static double MCCScoreCalculation(int tps, int tns, int fps, int fns)
        {
            var mccScore = ((tps * tns - fps * fns) * 1.0) / Math.Sqrt(((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)) * 1.0);

            return mccScore;
        }
    }
}
