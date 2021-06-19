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
        double bootstrapMCCScoreStandarddDev, bootstrapAccuracyStandarddDev, bootstrapPrecisionStandarddDev, bootstrapRecallStandarddDev, bootstrapF1ScoresStandardDev = 0.0;
        double bootstrapMCCScoreMean, bootstrapAccuracyMean, bootstrapPrecisionMean, bootstrapRecallMean, bootstrapF1ScoresMean = 0.0;

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
                var f1Scores = new List<double>(classificationMatrix.Count);

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
                    var f1Score = PerformanceMetrics.F1ScoreCalculation(tps, tns, fps, fns);
                    f1Scores.Add(f1Score);
                }

                this.bootstrapMCCScoreStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(mccScores.Where(a => !Double.IsNaN(a)));
                this.bootstrapMCCScoreMean = MathNet.Numerics.Statistics.Statistics.Mean(mccScores.Where(a => !Double.IsNaN(a)));
                this.bootstrapPrecisionStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(precisions.Where(a => !Double.IsNaN(a)));
                this.bootstrapPrecisionMean = MathNet.Numerics.Statistics.Statistics.Mean(precisions.Where(a => !Double.IsNaN(a)));
                this.bootstrapAccuracyStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(accuracies.Where(a => !Double.IsNaN(a)));
                this.bootstrapAccuracyMean = MathNet.Numerics.Statistics.Statistics.Mean(accuracies.Where(a => !Double.IsNaN(a)));
                this.bootstrapRecallStandarddDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(recalls.Where(a => !Double.IsNaN(a)));
                this.bootstrapRecallMean = MathNet.Numerics.Statistics.Statistics.Mean(recalls.Where(a => !Double.IsNaN(a)));
                this.bootstrapF1ScoresStandardDev = MathNet.Numerics.Statistics.Statistics.StandardDeviation(f1Scores.Where(a => !Double.IsNaN(a)));
                this.bootstrapF1ScoresMean = MathNet.Numerics.Statistics.Statistics.Mean(f1Scores.Where(a => !Double.IsNaN(a)));
            }
        }

        public double Accuracy {
            get
            {
                return PerformanceMetrics.AccuracyCalculation(this.tps, this.tns, this.fps, this.fns);
            }
        }

        public double BootstrapAccuracyMean
        {
            get
            {
                return this.bootstrapAccuracyMean;
            }
        }

        public double BootstrapAccuracyStandardDeviation
        {
            get
            {
                return this.bootstrapAccuracyStandarddDev;
            }
        }

        public double F1Score
        {
            get
            {
                return PerformanceMetrics.F1ScoreCalculation(this.tps, this.tns, this.fps, this.fns);
            }
        }

        public double BootstrapF1ScoreMean
        {
            get
            {
                return this.bootstrapF1ScoresMean;
            }
        }

        public double BootstrapF1ScoreStandardDeviation
        {
            get
            {
                return this.bootstrapF1ScoresStandardDev;
            }
        }

        public double MCCScore
        {
            get
            {
                return PerformanceMetrics.MCCScoreCalculation(this.tps, this.tns, this.fps, this.fns);
            }
        }

        public double BootstrapMCCScoreMean
        {
            get
            {
                return this.bootstrapMCCScoreMean;
            }
        }

        public double BootstrapMCCScoreStandardDeviation
        {
            get
            {
                return this.bootstrapMCCScoreStandarddDev;
            }
        }

        public double Precsion
        {
            get
            {
                return PerformanceMetrics.PrecisionCalculation(this.tps, this.tns, this.fps, this.fns);
            }
        }

        public double BootstrapPrecisionMean
        {
            get
            {
                return this.bootstrapPrecisionMean;
            }
        }

        public double BootstrapPrecisionStandardDeviation
        {
            get
            {
                return this.bootstrapPrecisionStandarddDev;
            }
        }

        public double Recall
        {
            get
            {
                return PerformanceMetrics.RecallCalculation(this.tps, this.tns, this.fps, this.fns);
            }
        }

        public double BootStrapRecallMean
        {
            get
            {
                return this.bootstrapRecallMean;
            }
        }

        public double BootStrapRecallStandardDeviation
        {
            get
            {
                return this.bootstrapRecallStandarddDev;
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
            var mccNumerator = tps * tns - fps * fns;

            var mccDenominator = Math.Sqrt(
                1.0 * (tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)
            );

            var mccScore = mccNumerator / mccDenominator;

            return mccScore;
        }

        public static double F1ScoreCalculation(int tps, int tns, int fps, int fns)
        {
            var precision = PerformanceMetrics.PrecisionCalculation(tps, tns, fps, fns);
            var recall = PerformanceMetrics.RecallCalculation(tps, tns, fps, fns);

            var f1Score = 2.0 * (precision * recall) / (precision + recall);

            return f1Score;
        }

    }
}
