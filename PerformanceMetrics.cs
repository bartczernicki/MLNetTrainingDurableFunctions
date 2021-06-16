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

        public PerformanceMetrics(List<string> performanceMatrix)
        {
            this.performanceMatrix = performanceMatrix;

            // Calculate counts of each metric
            this.tps = performanceMatrix.Count(t => t == "TP");
            this.tns = performanceMatrix.Count(t => t == "TN");
            this.fps = performanceMatrix.Count(t => t == "FP");
            this.fns = performanceMatrix.Count(t => t == "FN");
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
