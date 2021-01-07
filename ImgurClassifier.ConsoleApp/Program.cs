using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using ImgurClassifier.Model.DataModels;
using System.Collections.Generic;
using Newtonsoft.Json;
using System.Text;

namespace ImgurClassifier.ConsoleApp
{
    internal class Program
    {
        //Machine Learning model to train, later load and use for predictions
        private const string MODEL_FILEPATH = @"../../../../ImgurClassifier.Model/MLModel.zip";

        // Output logs
        private const string LOG_FILEPATH = @"../../../../ImgurClassifier.Model/log.txt";
        private static TextWriter logFs;
        private static readonly Object writerLock = new Object();

        private static void Main(string[] args)
        {

            Console.WriteLine("\n\n=============== Starting imgur classifier sample  ===============");

            using (logFs = File.AppendText(LOG_FILEPATH))
            {
                MLContext mlContext = new MLContext();

                mlContext.Log += ConsoleLogger;
                mlContext.Log += FileLogger;

                Dictionary<string, string> datasetSplits = DatasetDownloader.DatasetDownloader.GetDataset();
                string trainFileName = datasetSplits["train"];
                var validFileName = datasetSplits["valid"];
                var testFileName = datasetSplits["test"];

                // Training code used by ML.NET CLI and AutoML to generate the model
                File.Delete(MODEL_FILEPATH);
                ModelBuilder.CreateModel(mlContext, LogToFile);

                ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(MODEL_FILEPATH), out DataViewSchema inputSchema);

                var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

                // Create sample data to do a single prediction with it 
                ModelInput sampleData = CreateSingleDataSample(mlContext, testFileName);

                // Try a single prediction
                ModelOutput predictionResult = predEngine.Predict(sampleData);

                Console.WriteLine($"\nSample data:\n {JsonConvert.SerializeObject(sampleData, Formatting.Indented)}");
                Console.WriteLine($"Single Prediction --> Actual value: {sampleData.Label} | Predicted value: {predictionResult.Prediction} | Predicted scores: [{String.Join(",", predictionResult.Score)}]");

                Console.WriteLine("=============== Done with sample ===============");
                Console.WriteLine("=============== End of process, hit any key to finish ===============");
                Console.ReadKey();
            }
        }

        // Method to load single row of data to try a single prediction
        // You can change this code and create your own sample data here (Hardcoded or from any source)
        private static ModelInput CreateSingleDataSample(MLContext mlContext, string dataFilePath)
        {
            // Read dataset to get a single row for trying a prediction          
            IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: dataFilePath,
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Here (ModelInput object) you could provide new test data, hardcoded or from the end-user application, instead of the row from the file.
            ModelInput sampleForPrediction = mlContext.Data.CreateEnumerable<ModelInput>(dataView, false)
                                                                        .First();
            return sampleForPrediction;
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        private static void ConsoleLogger(object sender, LoggingEventArgs e)
        {
            if ((e.Kind == Microsoft.ML.Runtime.ChannelMessageKind.Error || e.Kind == Microsoft.ML.Runtime.ChannelMessageKind.Warning) && !e.RawMessage.StartsWith("Encountered imag"))
            {
                Console.WriteLine($"{DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss")} {e.Message}");
            }
        }

        private static void FileLogger(object sender, LoggingEventArgs e)
        {
            if ((e.Kind == Microsoft.ML.Runtime.ChannelMessageKind.Error || e.Kind == Microsoft.ML.Runtime.ChannelMessageKind.Warning || e.Kind == Microsoft.ML.Runtime.ChannelMessageKind.Info || (e.Source == "AutoML" && !e.RawMessage.StartsWith("[Source="))) && !e.RawMessage.StartsWith("Encountered imag"))
            {
                LogToFile($"{DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss")} {e.Message}", false);
            }
        }


        private static void LogToFile(string value)
        {
            LogToFile(value, true);
        }

        private static void LogToFile(string value, bool echoToConsole)
        {
            if (echoToConsole)
                Console.WriteLine(value);

            lock (writerLock)
            {
                logFs.WriteLine(value);
                logFs.Flush();
            }
        }

    }

}
