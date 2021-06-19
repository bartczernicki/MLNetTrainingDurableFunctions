using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.DurableTask;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.Extensions.Logging;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.Azure.Storage.Blob;
using System.Runtime.Caching;
using System;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Azure.Cosmos.Table;
using MLNetTrainingDurableFunctions.Tables;

namespace MLNetTrainingDurableFunctions
{
    public static class FuncOrch
    {
        static MemoryCache memoryCacheTest = new MemoryCache("TrainingData");

        // Info Variables
        private static int baseballPlayersCount = 0;
        private static List<MLBBaseballBatter> baseBallBatters;
        private static List<string> featureTargetLabels = new List<string>{ "InductedToHallOfFame", "OnHallOfFameBallot" };

        // Thread-safe ML Context
        // private static MLContext _mlContext = new MLContext(seed: 200);

        // BaseballDataService 
        private static BaseballDataSampleService baseBallDataService = BaseballDataSampleService.Instance;

        // Gam Algorithm HyperParameters
#if RELEASE
        private static int numberOfIterations = 1000, maximumBinCountPerFeature = 500;
        private static double learningRate = 0.001;
#else
        private static int numberOfIterations = 75, maximumBinCountPerFeature = 75;
        private static double learningRate = 0.05;
#endif

        // Model Features
        private static string[] featureColumns = new string[] {
            "YearsPlayed", "AB", "R", "H", "Doubles", "Triples", "HR", "RBI", "SB",
            "BattingAverage", "SluggingPct", "AllStarAppearances", "TB", "TotalPlayerAwards"
            // Other Features (Optional)
            /*, "MVPs", "TripleCrowns", "GoldGloves", "MajorLeaguePlayerOfTheYearAwards"*/
        };

        [FunctionName("BaseballFunc_HttpStart")]
        public static async Task<HttpResponseMessage> HttpStart(
        [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post")] HttpRequestMessage req,
        [DurableClient] IDurableOrchestrationClient starter,
        ILogger log)
        {
            // Function input comes from the request content.
            string instanceId = await starter.StartNewAsync("BaseballFunc_Orchestrator", null, input: "MLNet Training Job");

            log.LogInformation($"Started orchestration with ID = '{instanceId}'.");

            return starter.CreateCheckStatusResponse(req, instanceId);
        }

        [FunctionName("BaseballFunc_Orchestrator")]
        public static async Task<int> MLNetTrainingOrchestrator([OrchestrationTrigger] IDurableOrchestrationContext context, ILogger log)
        {
            var baseBallPlayers = await context.CallActivityAsync<List<MLBBaseballBatter>>("BaseballFunc_GetMLBBatters", null);

            var tasks = new List<Task<PredictionResult>>();

            // For Debug process, smaller list of players. For Release, process full list
#if RELEASE
            log.LogInformation($"Orchestrator - Release Mode (full data).");
#else
            baseBallPlayers = baseBallPlayers.Take(200).ToList();
#endif

            baseballPlayersCount = baseBallPlayers.Count;
            log.LogInformation($"Orchestrator - MLB Batters Count: {baseballPlayersCount}");

            foreach (var featureLabelTarget in featureTargetLabels)
            {
                foreach (var baseBallBatter in baseBallPlayers)
                {
                    // Add Train Model function activity
                    tasks.Add(context.CallActivityAsync<PredictionResult>("BaseballFunc_TrainModel",
                        input: new MLNetTrainingFunctionInput
                        {
                            FeatureLabelTarget = featureLabelTarget,
                            Batter = baseBallBatter
                        }));
                }
            }

            // Wait for all tasks to finish
            await Task.WhenAll(tasks);
            var predictionResults = tasks.Select(a => a.Result).ToList();

            // Calculate performance metrics
            var result = await context.CallActivityAsync<List<MLBBaseballBatter>>("BaseballFunc_CalculatePerformanceMetrics", predictionResults);

            return 1;
        }

        [FunctionName("BaseballFunc_GetMLBBatters")]
        public static List<MLBBaseballBatter> MLNetTrainingGetMLBBatters([ActivityTrigger] string name, ILogger log)
        {
            log.LogInformation($"GetMLBBatters - Getting MLB Baseball Batters...");

            // Retrieve the Baseball Data
            var baseBallData = baseBallDataService.GetTrainingBaseballData();
            baseBallBatters = baseBallData.Result;

            log.LogInformation($"GetMLBBatters - Finished Getting MLB Baseball Batters: {baseBallBatters}.");

            return baseBallBatters;
        }

        [FunctionName("BaseballFunc_TrainModel")]
        public static PredictionResult MLNetTrainingTrainModel([ActivityTrigger] MLNetTrainingFunctionInput mlNetTrainingFunctionInput, ILogger log)
        {
            var featureLabelTarget = mlNetTrainingFunctionInput.FeatureLabelTarget;

            log.LogInformation($"TrainModel - {featureLabelTarget} - Processing MLB Batter {mlNetTrainingFunctionInput.Batter.ID} - {mlNetTrainingFunctionInput.Batter.FullPlayerName}.");

            var _mlContext = new MLContext(seed: 200);

            var baseballBatters = BaseballDataSampleService.Instance.SampleBaseBallData;

            if (baseballBatters != null)
            {
                // Load for ML.NET DataView, excluding the one MLB Batter
                var baseBallPlayerDataView = _mlContext.Data.LoadFromEnumerable<MLBBaseballBatter>(
                    baseballBatters.Where(a => a.ID != mlNetTrainingFunctionInput.Batter.ID)
                    );

                EstimatorChain<Microsoft.ML.Transforms.NormalizingTransformer> baselineTransform = _mlContext.Transforms.Concatenate("FeaturesBeforeNormalization", featureColumns)
                    .Append(_mlContext.Transforms.NormalizeMinMax("Features", "FeaturesBeforeNormalization"));


                // Build simple data pipeline, with LONGER learning
                var learingPipeline =
                    baselineTransform.Append(
                    _mlContext.BinaryClassification.Trainers.Gam(labelColumnName: featureLabelTarget, numberOfIterations: numberOfIterations, learningRate: learningRate, maximumBinCountPerFeature: maximumBinCountPerFeature)
                    );

                log.LogInformation($"TrainModel - {featureLabelTarget} - Created Pipeline validating Gam Hyperparameters: Iterations: {numberOfIterations} LearningRate:{learningRate} MaxBinCountPerFeature:{maximumBinCountPerFeature}");
                log.LogInformation($"TrainModel - {featureLabelTarget} - Created Pipeline validating MLB Batter {mlNetTrainingFunctionInput.Batter.ID} - {mlNetTrainingFunctionInput.Batter.FullPlayerName}.");

                var model = learingPipeline.Fit(baseBallPlayerDataView);
                log.LogInformation($"TrainModel - {featureLabelTarget} Created Model validating MLB Batter {mlNetTrainingFunctionInput.Batter.ID} - {mlNetTrainingFunctionInput.Batter.FullPlayerName}.");

                var predictionEngine = _mlContext.Model.CreatePredictionEngine<MLBBaseballBatter, MLBHOFPrediction>(model);
                var prediction = predictionEngine.Predict(mlNetTrainingFunctionInput.Batter);
                log.LogInformation($"TrainModel - {featureLabelTarget} Prediction for MLB Batter {mlNetTrainingFunctionInput.Batter.ID} - {mlNetTrainingFunctionInput.Batter.FullPlayerName} || {prediction.Probability}");

                string predictionClass = string.Empty;

                if (featureLabelTarget == "OnHallOfFameBallot")
                {
                    if (mlNetTrainingFunctionInput.Batter.OnHallOfFameBallot && prediction.Probability >= 0.5f)
                    {
                        predictionClass = "TP";
                    }
                    else if (mlNetTrainingFunctionInput.Batter.OnHallOfFameBallot && prediction.Probability < 0.5f)
                    {
                        predictionClass = "FN";
                    }
                    else if (!mlNetTrainingFunctionInput.Batter.OnHallOfFameBallot && prediction.Probability < 0.5f)
                    {
                        predictionClass = "TN";
                    }
                    else if (!mlNetTrainingFunctionInput.Batter.OnHallOfFameBallot && prediction.Probability >= 0.5f)
                    {
                        predictionClass = "FP";
                    }
                }
                else
                {
                    if (mlNetTrainingFunctionInput.Batter.InductedToHallOfFame && prediction.Probability >= 0.5f)
                    {
                        predictionClass = "TP";
                    }
                    else if (mlNetTrainingFunctionInput.Batter.InductedToHallOfFame && prediction.Probability < 0.5f)
                    {
                        predictionClass = "FN";
                    }
                    else if (!mlNetTrainingFunctionInput.Batter.InductedToHallOfFame && prediction.Probability < 0.5f)
                    {
                        predictionClass = "TN";
                    }
                    else if (!mlNetTrainingFunctionInput.Batter.InductedToHallOfFame && prediction.Probability >= 0.5f)
                    {
                        predictionClass = "FP";
                    }
                }

                var predictionResult = new PredictionResult { FeatureLabelTarget = featureLabelTarget, PredictionClass = predictionClass, Probability = prediction.Probability };

                log.LogInformation($"TrainModel - {featureLabelTarget} - Prediction for MLB Batter {mlNetTrainingFunctionInput.Batter.ID} - {mlNetTrainingFunctionInput.Batter.FullPlayerName} ||" +
                    $" OnHallOfFameBallot: {mlNetTrainingFunctionInput.Batter.OnHallOfFameBallot}, Prediction Result: {predictionClass}");

                return predictionResult;
            }
            else
            {
                log.LogInformation($"TrainModel - Batters data NULL.");
                var predictionResult = new PredictionResult { FeatureLabelTarget = featureLabelTarget, PredictionClass = "NA", Probability = Double.NaN };

                return predictionResult;
            }
        }

        [FunctionName("BaseballFunc_CalculatePerformanceMetrics")]
        public static async Task<string> MLNetTrainingCalculatePerformanceMetrics([ActivityTrigger] List<PredictionResult> predictionResults,
            [Table("BaseballMlNetTrainingPerformanceMetrics")] CloudTable performanceMetricsTable,
            ILogger log)
        {
            log.LogInformation($"CalculatePerformanceMetrics - Calculating Performance Results...");

            var featureTargetLabels = predictionResults.Select(a => a.FeatureLabelTarget).Distinct().ToList();

            foreach (var featureTargetLabel in featureTargetLabels)
            {
                var classficationMatrix = predictionResults.Where(a => a.FeatureLabelTarget == featureTargetLabel).Select(m => m.PredictionClass).ToList();

                // Calculate performance metrics & bootstrap standard deviations to calculate confidence intervals
                var metrics = new PerformanceMetrics(classficationMatrix, true, 500);

                var performanceMetricsEntity = new TrainingJobPerformanceMetrics(featureTargetLabel, "Gam");
                performanceMetricsEntity.HyperParameters = $"Iterations: {numberOfIterations} LearningRate:{learningRate} MaxBinCountPerFeature: {maximumBinCountPerFeature}";
                performanceMetricsEntity.TruePositives = metrics.TruePositives;
                performanceMetricsEntity.TrueNegatives = metrics.TrueNegatives;
                performanceMetricsEntity.FalseNegatives = metrics.FalseNegatives;
                performanceMetricsEntity.FalsePositives = metrics.FalsePositives;
                performanceMetricsEntity.ValidationDataAccuracy = metrics.Accuracy;
                performanceMetricsEntity.ValidationDataPrecision = metrics.Precsion;
                performanceMetricsEntity.ValidationDataRecall = metrics.Recall;
                performanceMetricsEntity.ValidationDataMCCScore = metrics.MCCScore;
                performanceMetricsEntity.ValidationDataF1Score = metrics.F1Score;
                performanceMetricsEntity.BootstrapAccuracyStandardDeviation = metrics.BootstrapAccuracyStandardDeviation;
                performanceMetricsEntity.BootstrapPrecisionStandardDeviation = metrics.BootstrapPrecisionStandardDeviation;
                performanceMetricsEntity.BootstrapRecallStandardDeviation = metrics.BootStrapRecallStandardDeviation;
                performanceMetricsEntity.BootstrapF1ScoreStandardDeviation = metrics.BootstrapF1ScoreStandardDeviation;
                performanceMetricsEntity.BootstrapMCCScoreStandardDeviation = metrics.BootstrapMCCScoreStandardDeviation;
                performanceMetricsEntity.BootstrapAccuracyMean = metrics.BootstrapAccuracyMean;
                performanceMetricsEntity.BootstrapPrecisionMean = metrics.BootstrapPrecisionMean;
                performanceMetricsEntity.BootstrapRecallMean = metrics.BootStrapRecallMean;
                performanceMetricsEntity.BootstrapF1ScoreMean = metrics.BootstrapF1ScoreMean;
                performanceMetricsEntity.BootstrapMCCScoreMean = metrics.BootstrapMCCScoreMean;

                // Persist in Azure Table Storage
                var addEntryOperation = TableOperation.InsertOrReplace(performanceMetricsEntity);
                performanceMetricsTable.CreateIfNotExists();
                await performanceMetricsTable.ExecuteAsync(addEntryOperation);

                log.LogInformation($"Orchestrator - Predictions Matrix: TP:{metrics.TruePositives} TN:{metrics.TrueNegatives} FP:{metrics.FalsePositives} FN:{metrics.FalseNegatives}.");
                log.LogInformation($"Orchestrator - Performance Metrics: MCC Score:{metrics.MCCScore} Precision:{metrics.Precsion} Recall:{metrics.Recall}.");
            }

            return string.Empty;
        }
    }
}