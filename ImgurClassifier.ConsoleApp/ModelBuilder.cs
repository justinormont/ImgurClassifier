using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ImgurClassifier.Model.DataModels;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace ImgurClassifier.ConsoleApp
{
    public static class ModelBuilder
    {
        private static string modelFileName = @"../../../../ImgurClassifier.Model/MLModel.zip";

        static readonly string trainFileName;
        static readonly string validFileName;
        static readonly string testFileName;
        static readonly string basepath;

        static ModelBuilder()
        {
            Dictionary<string, string> datasetSplits = DatasetDownloader.DatasetDownloader.GetDataset();
            trainFileName = datasetSplits["train"];
            validFileName = datasetSplits["valid"];
            testFileName = datasetSplits["test"];
            basepath = datasetSplits["basepath"];
        }

        // Create MLContext to be shared across the model creation workflow objects 
        // Set a random seed for repeatable/deterministic results across multiple trainings.
        private static MLContext mlContext = new MLContext(seed: 1);

        public static void CreateModel()
        {
            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: trainFileName,
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: true,
                                            allowSparse: false);

            IDataView validationDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: validFileName,
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: true,
                                            allowSparse: false);

            IDataView testDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: testFileName,
                                            hasHeader: true,
                                            separatorChar: '\t',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Build training pipeline
            //IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipelineUsingAutoML(mlContext, trainingDataView, validationDataView);

            // Train Model
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            // Evaluate quality of Model
            EvaluateModel(mlContext, mlModel, testDataView);

            // Save model
            SaveModel(mlContext, mlModel, modelFileName, trainingDataView.Schema);

            // Print model weights of a proxy model
            PrintProxyModelWeights(mlContext, trainingDataView, trainingPipeline);
        }

        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                                                // Reset the Weight column to "1.0"
                                                .Append(mlContext.Transforms.Expression("Weight", "x : 1.0f", new[] { "Weight"} ))

                                                // Numeric features
                                                .Append(mlContext.Transforms.Concatenate("FeaturesNumeric", new[] { "tagAvgFollowers", "tagSumFollowers", "tagAvgTotalItems", "tagSumTotalItems", "imagesCount" }))

                                                // Categorical features
                                                .Append(mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair("img1Type", "img1Type"), new InputOutputColumnPair("img2Type", "img2Type"), new InputOutputColumnPair("img3Type", "img3Type") }))
                                                .Append(mlContext.Transforms.Concatenate("FeaturesCategorical", new[] { "img1Type", "img2Type", "img3Type" }))

                                                // Text features
                                                .Append(mlContext.Transforms.Text.FeaturizeText("imgDesc_tf", null, new[] { "img1Desc", "img2Desc", "img3Desc" }))
                                                .Append(mlContext.Transforms.Text.FeaturizeText("tags_tf", "tags"))
                                                .Append(mlContext.Transforms.Text.FeaturizeText("title_tf", "title"))
                                                .Append(mlContext.Transforms.Concatenate("FeaturesText", new[] { "title_tf", "tags_tf", "imgDesc_tf" }))

                                                // Model stacking using an Averaged Perceptron on the text features
                                                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "FeaturesText", numberOfIterations: 10), labelColumnName: "Label"))
                                                .Append(mlContext.Transforms.Concatenate("FeaturesStackedAPOnText", new[] { "Score" })) // Score column from the stacked trainer


                                                // Image features
                                                .Append(mlContext.Transforms.LoadImages("ImageObject", basepath, "img1FileName"))
                                                .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
                                                .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
                                                .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage1", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"))
                                                .Append(mlContext.Transforms.Concatenate("FeaturesImage", new[] { "FeaturesImage1" }))
                                                .AppendCacheCheckpoint(env: mlContext) // Cache checkpoint since the DnnFeaturizeImage is slow

                                                // Model stacking using a logistic regression on the image features to re-learn the output layer of the ResNet model 
                                                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "FeaturesImage"))
                                                .Append(mlContext.Transforms.Concatenate("FeaturesStackedLROnImages", new[] { "Score" })) // Score column from the stacked trainer

                                                .Append(mlContext.Transforms.Concatenate("Features", new[] {
                                                    "FeaturesNumeric",
                                                    "FeaturesCategorical",
                                                    "FeaturesText",
                                                    "FeaturesStackedAPOnText",
                                                    //"FeaturesImage",
                                                    "FeaturesStackedLROnImages",
                                                }))
                                                .AppendCacheCheckpoint(mlContext);

            // Set the training algorithm 
            //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features", numberOfIterations: 10), labelColumnName: "Label")
            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"), labelColumnName: "Label")
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        public static IEstimator<ITransformer> BuildTrainingPipelineUsingAutoML(MLContext mlContext, IDataView trainDataView, IDataView validationDataView)
        {

            Console.WriteLine("\n=============== Training Stacked AutoML model on Image features ===============");
            var progressHandler2 = new MulticlassExperimentProgressHandler();
            var experimentSettings2 = new MulticlassExperimentSettings
            {
                MaxExperimentTimeInSeconds = 3600,
                OptimizingMetric = MulticlassClassificationMetric.LogLossReduction,
            };
            // Set the column purposes for a subset of columns; the rest are auto-inferred
            ColumnInformation columnInformation2 = new ColumnInformation();
            columnInformation2.ExampleWeightColumnName = "Weight";
            columnInformation2.LabelColumnName = "Label";
            var columnsToIgnore2 = new List<string>() {
                "Name", "title", "tags", "tagCount", "tagAvgFollowers", "tagSumFollowers", "tagAvgTotalItems", "tagSumTotalItems", "imagesCount", "img1Uri",
                "img1Desc", "img1Type", "img1FileName", "img2Uri", "img2Desc", "img2Type", "img2FileName", "img3Uri", "img3Desc", "img3Type", "img3FileName",
                //"ImageObject", "Pixels", "FeaturesImage1" // The intermediate columns added from the preFeaturizer
                // todo: file bug to have AutoML check for columns to set the purpose of after the preFeaturiizer runs.
            };
            columnsToIgnore2.ForEach(a => columnInformation2.IgnoredColumnNames.Add(a));

            var preFeaturizer = mlContext.Transforms.LoadImages("ImageObject", basepath, "img1FileName")
                .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
                .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
                .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage1", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"))
                .Append(mlContext.Transforms.Concatenate("FeaturesImage", new[] { "FeaturesImage1" }))
                .Append(mlContext.Transforms.DropColumns(new [] { "ImageObject", "Pixels", "FeaturesImage1" })) // Drop intermediate featues so AutoML won't read them and be slowed down
                .AppendCacheCheckpoint(env: mlContext); // Cache checkpoint since the DnnFeaturizeImage is slow

            ExperimentResult<MulticlassClassificationMetrics> experimentResult2 = mlContext.Auto()
                .CreateMulticlassClassificationExperiment(experimentSettings2)
                .Execute(
                    trainData: mlContext.Data.TakeRows(mlContext.Data.ShuffleRows(input: trainDataView, shufflePoolSize: 5000), 100),
                    validationData: validationDataView,
                    progressHandler: progressHandler2,
                    columnInformation: columnInformation2,
                    preFeaturizer: preFeaturizer);
            Console.WriteLine("=============== Finished training AutoML model ===============");



            Console.WriteLine("\n=============== Training Stacked AutoML model on Text, Categorical, Numeric features ===============");
            var progressHandler = new MulticlassExperimentProgressHandler();
            var experimentSettings = new MulticlassExperimentSettings
            {
                MaxExperimentTimeInSeconds = 3600,
                OptimizingMetric = MulticlassClassificationMetric.LogLossReduction,
            };

            // Set the column purposes for a subset of columns; the rest are auto-inferred
            ColumnInformation columnInformation = new ColumnInformation();
            columnInformation.ExampleWeightColumnName = "Weight";
            columnInformation.LabelColumnName = "Label";
            var columnsToIgnore = new List<string>() { "Name", "img1Uri", "img1FileName", "img2Uri", "img2FileName", "img3Uri", "img3FileName" };
            columnsToIgnore.ForEach(a => columnInformation.IgnoredColumnNames.Add(a));

            ExperimentResult<MulticlassClassificationMetrics> experimentResult1 = mlContext.Auto()
                .CreateMulticlassClassificationExperiment(experimentSettings)
                .Execute(
                    trainData: trainDataView,
                    validationData: validationDataView,
                    progressHandler: progressHandler,
                    columnInformation: columnInformation);
            Console.WriteLine("=============== Finished training AutoML model ===============");


            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                // Reset the Weight column to "1.0"
                .Append(mlContext.Transforms.Expression("Weight", "x : 1.0f", new[] { "Weight" }))

                // Numeric features
                .Append(mlContext.Transforms.Concatenate("FeaturesNumeric", new[] { "tagAvgFollowers", "tagSumFollowers", "tagAvgTotalItems", "tagSumTotalItems", "imagesCount" }))

                // Categorical features
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair("img1Type", "img1Type"), new InputOutputColumnPair("img2Type", "img2Type"), new InputOutputColumnPair("img3Type", "img3Type") }))
                .Append(mlContext.Transforms.Concatenate("FeaturesCategorical", new[] { "img1Type", "img2Type", "img3Type" }))

                // Text features
                .Append(mlContext.Transforms.Text.FeaturizeText("imgDesc_tf", null, new[] { "img1Desc", "img2Desc", "img3Desc" }))
                .Append(mlContext.Transforms.Text.FeaturizeText("tags_tf", "tags"))
                .Append(mlContext.Transforms.Text.FeaturizeText("title_tf", "title"))
                .Append(mlContext.Transforms.Concatenate("FeaturesText", new[] { "title_tf", "tags_tf", "imgDesc_tf" }))

                // Model stacking using an Averaged Perceptron on the text features
                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "FeaturesText", numberOfIterations: 10), labelColumnName: "Label"))
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedAPOnText", new[] { "Score" })) // Score column from the stacked trainer


                // Image features
                .Append(mlContext.Transforms.LoadImages("ImageObject", basepath, "img1FileName"))
                .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
                .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
                .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage1", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"))
                .Append(mlContext.Transforms.Concatenate("FeaturesImage", new[] { "FeaturesImage1" }))
                .AppendCacheCheckpoint(env: mlContext) // Cache checkpoint since the DnnFeaturizeImage is slow

                // Model stacking using a logistic regression on the image features to re-learn the output layer of the ResNet model 
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "FeaturesImage"))
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedLROnImages", new[] { "Score" })) // Score column from the stacked trainer

                // Model stacking using the previously learned AutoML model on text, categorical, and numeric features (note: the AutoML estimator chain may mask existing columns)
                .Append(experimentResult1.BestRun.Estimator)
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedAutoMLOnTextCatNum", new[] { "Score" })) // Score column from the stacked trainer

                // Model stacking using the previously learned AutoML model on Imge features (note: the AutoML estimator chain may mask existing columns)
                .Append(experimentResult2.BestRun.Estimator)
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedAutoMLOnImages", new[] { "Score" })) // Score column from the stacked trainer

                .Append(mlContext.Transforms.Concatenate("Features", new[] {
                    "FeaturesNumeric",
                    "FeaturesCategorical",
                    "FeaturesText",
                    "FeaturesStackedAPOnText",
                    //"FeaturesImage",
                    "FeaturesStackedLROnImages",
                    "FeaturesStackedAutoMLOnTextCatNum",
                    "FeaturesStackedAutoMLOnImages",
                }))
                .AppendCacheCheckpoint(mlContext);

            // Set the training algorithm 
            //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features", numberOfIterations: 10), labelColumnName: "Label")
            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"), labelColumnName: "Label")
                                      .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of training process ===============\n");
            return model;
        }

        private static void EvaluateModel(MLContext mlContext, ITransformer mlModel, IDataView testDataView)
        {
            // Evaluate the model and show accuracy stats
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            IDataView predictions = mlModel.Transform(testDataView);

            // Obtuse method of getting the number of classes
            VBuffer<ReadOnlyMemory<char>> classNamesVBuffer = default;
            predictions.Schema["Score"].GetSlotNames(ref classNamesVBuffer);
            var numClasses = classNamesVBuffer.Length;
            string[] classNames = classNamesVBuffer.DenseValues().Select(a => a.ToString()).ToArray();

            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score",  topKPredictionCount: numClasses); // Todo: fix bug to allow for `topKPredictionCount: Int32.MaxValue` 
            ConsoleHelper.PrintMulticlassClassificationMetrics(metrics, classNames);
        }

        private static void PrintProxyModelWeights(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {

            /*
            // Train a proxy model
            var pipeline = trainingPipeline.Append(mlContext.Regression.Trainers.FastForest(
                options: new FastForestRegressionTrainer.Options()
                {
                    FeatureColumnName = "Features",
                    LabelColumnName = "Score",
                    ExampleWeightColumnName = "Weight",
                    ShuffleLabels = true,
                }));

            */

            var trainerOptions = new FastForestRegressionTrainer.Options
            {
                //FeatureFirstUsePenalty = 0.1,
                NumberOfLeaves = 20,
                FeatureFraction = 0.7,
                NumberOfTrees = 500,
                LabelColumnName = "FloatLabel",
                FeatureColumnName = "Features",
                //ExampleWeightColumnName = "Weight",

                // Shuffle the label ordering before each tree is learned.
                // Needed when running a multi-class dataset as regression.
                ShuffleLabels = true,
            };

            // Define the tree-based featurizer's configuration.
            var options = new FastForestRegressionFeaturizationEstimator.Options
            {
                InputColumnName = "Features",
                TreesColumnName = "FeaturesTreeFeatTrees",
                LeavesColumnName = "FeaturesTreeFeatLeaves",
                PathsColumnName = "FeaturesTreeFeatPaths",
                TrainerOptions = trainerOptions
            };

            Action<RowWithKey, RowWithFloat> actionConvertKeyToFloat = (RowWithKey rowWithKey, RowWithFloat rowWithFloat) =>
            {
                rowWithFloat.FloatLabel = rowWithKey.Label == 0 ? float.NaN : rowWithKey.Label - 1;
            };

            // Convert the Key
            var pipeline = trainingPipeline
                // Convert the Key type to a Float (so we can use a regression trainer)
                .Append(mlContext.Transforms.CustomMapping(actionConvertKeyToFloat, "Label"))

                // Train a FastForestRegression model
                .Append(mlContext.Transforms.FeaturizeByFastForestRegression(options));

            Console.WriteLine("=============== Training proxy model ===============");

            // Fit this pipeline to the training data.
            var model = pipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of proxy training process ===============\n");

            // Get the feature importance based on the information gain used during training.
            VBuffer<float> weights = default;
            model.LastTransformer.Model.GetFeatureWeights(ref weights);
            float[] weightsValues = weights.DenseValues().ToArray();

            // Get the name of the features (slot names)
            var output = model.Transform(trainingDataView);
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            output.Schema["Features"].GetSlotNames(ref slotNames);

            // Sort to place the most important features first
            IEnumerable<string> slotWeightText = slotNames.Items()
                .Select((x, i) => ($"{x.Value}", (float)weightsValues[i], (float)Math.Abs(weightsValues[i])))
                .Where(t => t.Item3 > 0)
                .OrderByDescending(t => t.Item3)
                .Take(100)
                .Select(x => $"{x.Item1}: {x.Item2}");

            Console.WriteLine("\nFeature importance:");
            Console.WriteLine(String.Join("\n", slotWeightText));
        }

        private class RowWithKey
        {
            [KeyType(99999)]
            public uint Label { get; set; }
        }

        private class RowWithFloat
        {
            public float FloatLabel { get; set; }
        }

        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            // Save/persist the trained model to a .ZIP file
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        
        
    }

    /// <summary>
    /// Progress handler that AutoML will invoke after each model it produces and evaluates.
    /// </summary>
    public class MulticlassExperimentProgressHandler : IProgress<RunDetail<MulticlassClassificationMetrics>>
    {
        private int _iterationIndex;

        public void Report(RunDetail<MulticlassClassificationMetrics> iterationResult)
        {
            if (_iterationIndex++ == 0)
            {
                ConsoleHelper.PrintMulticlassClassificationMetricsHeader();
            }

            if (iterationResult.Exception != null)
            {
                ConsoleHelper.PrintIterationException(iterationResult.Exception);
            }
            else
            {
                ConsoleHelper.PrintIterationMetrics(_iterationIndex, iterationResult.TrainerName,
                    iterationResult.ValidationMetrics, iterationResult.RuntimeInSeconds);
            }
        }
    }

    public static class ConsoleHelper
    {
        private const int Width = 114;

        public static void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics, string[] classNames)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"Accuracy (micro-avg):              {metrics.MicroAccuracy:0.0000}   # 0..1, higher is better");
            Console.WriteLine($"Accuracy (macro):                  {metrics.MacroAccuracy:0.0000}   # 0..1, higher is better");
            Console.WriteLine($"Top-K accuracy:                    [{string.Join(", ", metrics?.TopKAccuracyForAllK?.Select(a => $"{a:0.0000}") ?? new string[] { "Set topKPredictionCount in evaluator to view" })}]   # 0..1, higher is better");
            Console.WriteLine($"Log-loss reduction:                {metrics.LogLossReduction:0.0000;-0.000}   # -Inf..1, higher is better");
            Console.WriteLine($"Log-loss:                          {metrics.LogLoss:0.0000}   # 0..Inf, lower is better");
            Console.WriteLine("\nPer class metrics");
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"LogLoss for class {i} ({classNames[i] + "):",-11}   {metrics.PerClassLogLoss[i]:0.0000}   # 0..Inf, lower is better");
            }
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"Precision for class {i} ({classNames[i] + "):",-11} {metrics.ConfusionMatrix.PerClassPrecision[i]:0.0000}   # 0..1, higher is better");
            }
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                Console.WriteLine($"Recall for class {i} ({classNames[i] + "):",-11}    {metrics.ConfusionMatrix.PerClassRecall[i]:0.0000}   # 0..1, higher is better");
            }
            Console.WriteLine();
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine($"************************************************************");
        }


        internal static void PrintIterationMetrics(int iteration, string trainerName, MulticlassClassificationMetrics metrics, double? runtimeInSeconds)
        {
            CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.MicroAccuracy ?? double.NaN,14:F4} {metrics?.MacroAccuracy ?? double.NaN,14:F4} {metrics?.LogLossReduction ?? double.NaN,17:F4} {runtimeInSeconds.Value,9:F1}", Width);
        }

        internal static void PrintIterationException(Exception ex)
        {
            Console.WriteLine($"Exception during AutoML iteration: {ex}");
        }

        internal static void PrintMulticlassClassificationMetricsHeader()
        {
            CreateRow($"{"",-4} {"Trainer",-35} {"MicroAccuracy",14} {"MacroAccuracy",14} {"LogLossReduction",17} {"Duration",9}", Width);
        }

        private static void CreateRow(string message, int width)
        {
            Console.WriteLine("|" + message.PadRight(width - 2) + "|");
        }

        public static void PrintMulticlassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            Console.WriteLine($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");
        }

        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }
    }
}
