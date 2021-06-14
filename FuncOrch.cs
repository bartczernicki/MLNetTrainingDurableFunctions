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

namespace MLNetTrainingDurableFunctions
{
    public static class FuncOrch
    {
        static MemoryCache memoryCacheTest = new MemoryCache("TrainingData");

        // Info Variables
        private static int baseballPlayersCount = 0;
        private static int numberOfCorrectPredictions = 0;
        private static List<MLBBaseballBatter> baseBallBatters;

        // Thread-safe ML Context
        // private static MLContext _mlContext = new MLContext(seed: 200);

        // BaseballDataService 
        private static BaseballDataSampleService baseBallDataService = BaseballDataSampleService.Instance;
        //private static DataViewSchema mlbBattersSchema;

        // Model Features
        private static string[] featureColumns = new string[] {
            "YearsPlayed", "AB", "R", "H", "Doubles", "Triples", "HR", "RBI", "SB",
            "BattingAverage", "SluggingPct", "AllStarAppearances", "TB", "TotalPlayerAwards"
            // Other Features
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
            var baseBallPlayers = await context.CallActivityAsync<List<MLBBaseballBatter>>(
                "BaseballFunc_GetMLBBatters", null);

            var tasks = new List<Task<string>>();

#if RELEASE
            log.LogInformation($"Orchestrator - Release Mode (full data).");
#else
            baseBallPlayers = baseBallPlayers.Take(50).ToList();
#endif

            baseballPlayersCount = baseBallPlayers.Count;
            log.LogInformation($"Orchestrator - MLB Batters Count: {baseballPlayersCount}");

            foreach (var baseBallBatter in baseBallPlayers)
            {
                // Add Train Model function activity
                tasks.Add(context.CallActivityAsync<string>("BaseballFunc_TrainModel", input: baseBallBatter));
            }

            await Task.WhenAll(tasks);

            // Count up performance matrix
            var tps = tasks.Count(t => t.Result == "TP");
            var tns = tasks.Count(t => t.Result == "TN");
            var fps = tasks.Count(t => t.Result == "FP");
            var fns = tasks.Count(t => t.Result == "FN");
            var empty = tasks.Count(t => t.Result == string.Empty);

            // var correctPredictionsMessage = "Orchestrator - Number of Correct Predictions: " + numberOfCorrectPredictions.ToString();
            log.LogInformation($"Orchestrator - Prredictions Matrix: TP:{tps} TN:{tns} FP:{fps} FN:{fns}.");

            return numberOfCorrectPredictions;
        }

        [FunctionName("BaseballFunc_GetMLBBatters")]
        public static List<MLBBaseballBatter> MLNetTrainingGetMLBBatters([ActivityTrigger] string name, ILogger log)
        {
            log.LogInformation($"GetMLBBatters - Getting MLB Baseball Batters...");

            var baseBallData = baseBallDataService.GetTrainingBaseballData();
            baseBallBatters = baseBallData.Result;

            // Place in-memory Cache
            //CacheItemPolicy policy = new CacheItemPolicy();
            //memoryCacheTest.Set("MLBBaseballBatters", value: baseBallBatters, policy);

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
                // Build simple data pipeline
                var learingPipeline =
                    baselineTransform.Append(
                    _mlContext.BinaryClassification.Trainers.Gam(labelColumnName: labelColunmn, numberOfIterations: 250, learningRate: 0.01, maximumBinCountPerFeature: 200)
                    );
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
    }
}