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

        //[FunctionName("FuncOrch")]
        //public static async Task<List<string>> RunOrchestrator(
        //    [OrchestrationTrigger] IDurableOrchestrationContext context)
        //{
        //    var outputs = new List<string>();

        //    // Replace "hello" with the name of your Durable Activity Function.
        //    outputs.Add(await context.CallActivityAsync<string>("FuncOrch_Hello", "Tokyo"));
        //    outputs.Add(await context.CallActivityAsync<string>("FuncOrch_Hello", "Seattle"));
        //    outputs.Add(await context.CallActivityAsync<string>("FuncOrch_Hello", "London"));

        //    // returns ["Hello Tokyo!", "Hello Seattle!", "Hello London!"]
        //    return outputs;
        //}

        //[FunctionName("FuncOrch_Hello")]
        //public static string SayHello([ActivityTrigger] string name, ILogger log)
        //{
        //    log.LogInformation($"Saying hello to {name}.");
        //    return $"Hello {name}!";
        //}

        //[FunctionName("FuncOrch_HttpStart")]
        //public static async Task<HttpResponseMessage> HttpStart(
        //    [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post")] HttpRequestMessage req,
        //    [DurableClient] IDurableOrchestrationClient starter,
        //    ILogger log)
        //{
        //    // Function input comes from the request content.
        //    string instanceId = await starter.StartNewAsync("E2_BackupSiteContent", null);

        //    log.LogInformation($"Started orchestration with ID = '{instanceId}'.");

        //    return starter.CreateCheckStatusResponse(req, instanceId);
        //}

        //[FunctionName("E2_BackupSiteContent")]
        //public static async Task<long> Run([OrchestrationTrigger] IDurableOrchestrationContext backupContext)
        //{
        //    string rootDirectory = backupContext.GetInput<string>()?.Trim();
        //    if (string.IsNullOrEmpty(rootDirectory))
        //    {
        //        rootDirectory = Directory.GetParent(typeof(FuncOrch).Assembly.Location).FullName;
        //    }

        //    string[] files = await backupContext.CallActivityAsync<string[]>(
        //        "E2_GetFileList",
        //        rootDirectory);

        //    if (memoryCacheTest.GetCount() == 0)
        //    {
        //        CacheItemPolicy policy = new CacheItemPolicy();
        //        memoryCacheTest.Set("files", value: files, policy);
        //    }

        //    var tasks = new Task<long>[files.Length];
        //    for (int i = 0; i < files.Length; i++)
        //    {
        //        tasks[i] = backupContext.CallActivityAsync<long>(
        //            "E2_CopyFileToBlob",
        //            files[i]);
        //    }

        //    await Task.WhenAll(tasks);

        //    long totalBytes = tasks.Sum(t => t.Result);
        //    return totalBytes;
        //}

        //[FunctionName("E2_GetFileList")]
        //public static string[] GetFileList([ActivityTrigger] string rootDirectory, ILogger log)
        //{
        //    log.LogInformation($"Searching for files under '{rootDirectory}'...");
        //    string[] files = Directory.GetFiles(rootDirectory, "*", SearchOption.AllDirectories);
        //    log.LogInformation($"Found {files.Length} file(s) under {rootDirectory}.");

        //    return files;
        //}

        //[FunctionName("E2_CopyFileToBlob")]
        //public static async Task<long> CopyFileToBlob([ActivityTrigger] string filePath, Microsoft.Azure.WebJobs.Binder binder, ILogger log)
        //{
        //    string[] fileNames = memoryCacheTest["files"] as string[];
        //    log.LogInformation($"file names:'{fileNames.Length}'");

        //    long byteCount = new FileInfo(filePath).Length;

        //    // strip the drive letter prefix and convert to forward slashes
        //    string blobPath = filePath
        //        .Substring(Path.GetPathRoot(filePath).Length)
        //        .Replace('\\', '/');
        //    string outputLocation = $"backups/{blobPath}";

        //    //log.LogInformation($"Copying '{filePath}' to '{outputLocation}'. Total bytes = {byteCount}.");

        //    // copy the file contents into a blob
        //    using (Stream source = File.Open(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        //    using (Stream destination = await binder.BindAsync<CloudBlobStream>(
        //        new BlobAttribute(outputLocation, FileAccess.Write)))
        //    {
        //        await source.CopyToAsync(destination);
        //    }

        //    return byteCount;
        //}

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

            var tasks = new List<Task<bool>>();

#if RELEASE
            log.LogInformation($"Orchestrator - Release Mode (full data).");
#else
            baseBallPlayers = baseBallPlayers.Take(10).ToList();
#endif

            baseballPlayersCount = baseBallPlayers.Count;
            log.LogInformation($"Orchestrator - MLB Batters Count: {baseballPlayersCount}");

            foreach (var baseBallBatter in baseBallPlayers)
            {
                // Add Train Model function activity
                tasks.Add(context.CallActivityAsync<bool>("BaseballFunc_TrainModel", input: baseBallBatter));
            }

            await Task.WhenAll(tasks);

            numberOfCorrectPredictions = tasks.Count(t => t.Result == true);
            // var correctPredictionsMessage = "Orchestrator - Number of Correct Predictions: " + numberOfCorrectPredictions.ToString();
            log.LogInformation($"Orchestrator - Number of Correct Predictions: {numberOfCorrectPredictions} of {baseballPlayersCount}.");

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
        public static bool MLNetTrainingTrainModel([ActivityTrigger] MLBBaseballBatter batter, ILogger log)
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
                    _mlContext.BinaryClassification.Trainers.Gam(labelColumnName: labelColunmn, numberOfIterations: 200, learningRate: 0.01, maximumBinCountPerFeature: 150)
                    );
                log.LogInformation($"TrainModel - Created Pipeline validating MLB Batter {batter.ID} - {batter.FullPlayerName}.");

                var model = learingPipeline.Fit(baseBallPlayerDataView);
                log.LogInformation($"TrainModel - Created Model validating MLB Batter {batter.ID} - {batter.FullPlayerName}.");

                var predictionEngine = _mlContext.Model.CreatePredictionEngine<MLBBaseballBatter, MLBHOFPrediction>(model);
                var prediction = predictionEngine.Predict(batter);
                log.LogInformation($"TrainModel - Prediction for MLB Batter {batter.ID} - {batter.FullPlayerName} || {prediction.Probability}");

                var validPrediction = (batter.OnHallOfFameBallot && prediction.Probability >= 0.5f) ? true :
                    ((!batter.OnHallOfFameBallot && prediction.Probability < 0.5f) ? true : false);

                log.LogInformation($"TrainModel - Prediction for MLB Batter {batter.ID} - {batter.FullPlayerName} ||" +
                    $" OnHallOfFameBallot: {batter.OnHallOfFameBallot}, Does it Match pred: {validPrediction}");

                return validPrediction;
            }
            else
            {
                log.LogInformation($"TrainModel - Batters data NULL.");

                return false;
            }


        }
    }
}