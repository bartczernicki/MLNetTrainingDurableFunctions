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

        // Thread-safe ML Context
        // private static MLContext _mlContext = new MLContext(seed: 200);

        // BaseballDataService 
        private static BaseballDataSampleService baseBallDataService = BaseballDataSampleService.Instance;

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

            var tasks = new List<Task<string>>();

            // For Debug process, smaller list of players. For Release, process full list
#if RELEASE
            log.LogInformation($"Orchestrator - Release Mode (full data).");
#else
            baseBallPlayers = baseBallPlayers.Take(200).ToList();
#endif

            baseballPlayersCount = baseBallPlayers.Count;
            log.LogInformation($"Orchestrator - MLB Batters Count: {baseballPlayersCount}");

            foreach (var baseBallBatter in baseBallPlayers)
            {
                // Add Train Model function activity
                tasks.Add(context.CallActivityAsync<string>("BaseballFunc_TrainModel", input: baseBallBatter));
            }

            // Wait for all tasks to finish
            await Task.WhenAll(tasks);
            var trainModelResults = tasks.Select(a => a.Result).ToList();

            // Calculate performance metrics
            var result = await context.CallActivityAsync<List<MLBBaseballBatter>>("BaseballFunc_CalculatePerformanceMetrics", trainModelResults);

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
        public static string MLNetTrainingTrainModel([ActivityTrigger] MLBBaseballBatter batter, ILogger log)
        {
            log.LogInformation($"TrainModel - Processing MLB Batter {batter.ID} - {batter.FullPlayerName}.");

            var _mlContext = new MLContext(seed: 200);

            var baseballBatters = BaseballDataSampleService.Instance.SampleBaseBallData;

            if (baseballBatters != null)
            {
                // Load for ML.NET DataView, excluding the one MLB Batter
                var baseBallPlayerDataView = _mlContext.Data.LoadFromEnumerable<MLBBaseballBatter>(
                    baseballBatters.Where(a => a.ID != batter.ID)
                    );

                EstimatorChain<Microsoft.ML.Transforms.NormalizingTransformer> baselineTransform = _mlContext.Transforms.Concatenate("FeaturesBeforeNormalization", featureColumns)
                    .Append(_mlContext.Transforms.NormalizeMinMax("Features", "FeaturesBeforeNormalization"));

                var labelColunmn = "OnHallOfFameBallot";

#if RELEASE
                // Build simple data pipeline, with LONGER learning
                var learingPipeline =
                    baselineTransform.Append(
                    _mlContext.BinaryClassification.Trainers.Gam(labelColumnName: labelColunmn, numberOfIterations: 1000, learningRate: 0.001, maximumBinCountPerFeature: 500)
                    );
#else
                // Build complex data pipeline, with SHORTER learning
                var learingPipeline =
                    baselineTransform.Append(
                    _mlContext.BinaryClassification.Trainers.Gam(labelColumnName: labelColunmn, numberOfIterations: 75, learningRate: 0.05, maximumBinCountPerFeature: 75)
                    );
#endif

                log.LogInformation($"TrainModel - Created Pipeline validating MLB Batter {batter.ID} - {batter.FullPlayerName}.");

                var model = learingPipeline.Fit(baseBallPlayerDataView);
                log.LogInformation($"TrainModel - Created Model validating MLB Batter {batter.ID} - {batter.FullPlayerName}.");

                var predictionEngine = _mlContext.Model.CreatePredictionEngine<MLBBaseballBatter, MLBHOFPrediction>(model);
                var prediction = predictionEngine.Predict(batter);
                log.LogInformation($"TrainModel - Prediction for MLB Batter {batter.ID} - {batter.FullPlayerName} || {prediction.Probability}");

                string predictionResult = string.Empty;

                if (batter.OnHallOfFameBallot && prediction.Probability >= 0.5f)
                {
                    predictionResult = "TP";
                }
                else if (batter.OnHallOfFameBallot && prediction.Probability < 0.5f)
                {
                    predictionResult = "FN";
                }
                else if (!batter.OnHallOfFameBallot && prediction.Probability < 0.5f)
                {
                    predictionResult = "TN";
                }
                else if (!batter.OnHallOfFameBallot && prediction.Probability >= 0.5f)
                {
                    predictionResult = "FP";
                }

                log.LogInformation($"TrainModel - Prediction for MLB Batter {batter.ID} - {batter.FullPlayerName} ||" +
                    $" OnHallOfFameBallot: {batter.OnHallOfFameBallot}, Prediction Result: {predictionResult}");

                return predictionResult;
            }
            else
            {
                log.LogInformation($"TrainModel - Batters data NULL.");

                return string.Empty;
            }
        }

        [FunctionName("BaseballFunc_CalculatePerformanceMetrics")]
        public static async Task<string> MLNetTrainingCalculatePerformanceMetrics([ActivityTrigger] List<string> matrixPerformanceResults,
            [Table("BaseballMlNetTrainingPerformanceMetrics")] CloudTable performanceMetricsTable,
            ILogger log)
        {
            log.LogInformation($"CalculatePerformanceMetrics - Calculating Performance Results...");

            // Calculate performance metrics & bootstrap standard deviations to calculate confidence intervals
            var metrics = new PerformanceMetrics(matrixPerformanceResults, true, 500);

            var performanceMetricsEntity = new TrainingJobPerformanceMetrics("Test", "Gam");
            performanceMetricsEntity.HyperParameters = "75;0.05;75";
            performanceMetricsEntity.TruePositives = metrics.TruePositives;
            performanceMetricsEntity.TrueNegatives = metrics.TrueNegatives;
            performanceMetricsEntity.FalseNegatives = metrics.FalseNegatives;
            performanceMetricsEntity.FalsePositives = metrics.FalsePositives;
            performanceMetricsEntity.Accuracy = metrics.Accuracy;
            performanceMetricsEntity.Precision = metrics.Precsion;
            performanceMetricsEntity.Recall = metrics.Recall;
            performanceMetricsEntity.MCCScore = metrics.MCCScore;
            performanceMetricsEntity.F1Score = metrics.F1Score;
            performanceMetricsEntity.AccuracyBootStrapStandardDeviation = metrics.AccuracyBootStrapStandardDeviation;
            performanceMetricsEntity.PrecisionBootStrapStandardDeviation = metrics.PrecisionBootStrapStandardDeviation;
            performanceMetricsEntity.RecallBootStrapStandardDeviation = metrics.RecallBootStrapStandardDeviation;
            performanceMetricsEntity.MCCScoreBootStrapStandardDeviation = metrics.MCCScoreStandardDeviation;
            performanceMetricsEntity.F1ScoreBootStrapStandardDeviation = metrics.F1ScoreStandardDeviation;

            // Persist in Azure Table Storage
            var addEntryOperation = TableOperation.InsertOrReplace(performanceMetricsEntity);
            performanceMetricsTable.CreateIfNotExists();
            await performanceMetricsTable.ExecuteAsync(addEntryOperation);

            log.LogInformation($"Orchestrator - Predictions Matrix: TP:{metrics.TruePositives} TN:{metrics.TrueNegatives} FP:{metrics.FalsePositives} FN:{metrics.FalseNegatives}.");
            log.LogInformation($"Orchestrator - Performance Metrics: MCC Score:{metrics.MCCScore} Precision:{metrics.Precsion} Recall:{metrics.Recall}.");

            return string.Empty;
        }
    }
}