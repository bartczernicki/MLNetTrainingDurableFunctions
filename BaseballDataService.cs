using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace MLNetTrainingDurableFunctions
{
    public sealed class BaseballDataSampleService
    {
        private static readonly BaseballDataSampleService instance = new BaseballDataSampleService();

        private BaseballDataSampleService()
        {
            SampleBaseBallData = GetTrainingBaseballData().Result;
        }

        static BaseballDataSampleService()
        {
        }

        public static BaseballDataSampleService Instance
        {
            get
            {
                return instance;
            }
        }

        public IEnumerable<string> ReadLines(Func<Stream> streamProvider, Encoding encoding)
        {
            using (var stream = streamProvider())
            using (var reader = new StreamReader(stream, encoding))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    yield return line;
                }
            }
        }

        public static Stream GetBaseballDataStream()
        {
            var assembly = typeof(MLNetTrainingDurableFunctions.FuncOrch).Assembly;

            // var test = assembly.GetManifestResourceNames();
            // taskkill /IM dotnet.exe /F /T 2>nul 1>nul

            Stream resource = assembly.GetManifestResourceStream($"MLNetTrainingDurableFunctions.Data.MLBBaseballBattersFullTraining.csv");

            return resource;
        }

        public static async Task<Stream> GetBaseballDataStreamFromGitHub()
        {
            var httpClient = new HttpClient();
            var request = await httpClient.GetStreamAsync("https://raw.githubusercontent.com/bartczernicki/BaseballData/main/data/MLBBaseballBattersFullTraining.csv");

            return request;
        }

        public Task<List<MLBBaseballBatter>> GetTrainingBaseballData()
        {
            // Return sample baseball players (batters)
            // Mix of fictitious, active & retired players of all skills

            // Note: In a production system this service would load the list of batters
            // from distributed persisted storage, searched in information retrieval engine (i.e. Azure Search, Lucene),
            // a relational database etc.

            // Load MLB baseball batters from local CSV file

#if RELEASE
            var lines = ReadLines(() => GetBaseballDataStreamFromGitHub().Result, Encoding.UTF8);
#else
            var lines = ReadLines(() => GetBaseballDataStream(), Encoding.UTF8);
#endif

            var batters = lines
                        .Skip(1)
                        .Select(v => MLBBaseballBatter.FromCsv(v));

            return Task.FromResult(
                batters.ToList()
            ); ;
        }

        public List<MLBBaseballBatter> SampleBaseBallData
        {
            get;
            set;
        }

    }
}