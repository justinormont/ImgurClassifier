using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using ImgurClassifier.Model.DataModels;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;

namespace ImgurClassifier.ConsoleApp
{
    public static class ModelBuilder
    {
        private static readonly string modelFileName = @"../../../../ImgurClassifier.Model/MLModel.zip";
        private static readonly string trainFileName;
        private static readonly string validFileName;
        private static readonly string testFileName;
        private static readonly string basepath;

        static ModelBuilder()
        {
            Dictionary<string, string> datasetSplits = DatasetDownloader.DatasetDownloader.GetDataset();
            trainFileName = datasetSplits["train"];
            validFileName = datasetSplits["valid"];
            testFileName = datasetSplits["test"];
            basepath = datasetSplits["basepath"];
        }

        public static void CreateModel(MLContext mlContext)
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

            // Explicitly shuffle as the LoadFromTextFile doesn't allow shuffling for unknown reasons, and we can control the max pool size
            trainingDataView = mlContext.Data.ShuffleRows(trainingDataView, shufflePoolSize: 1_000_000);
            validationDataView = mlContext.Data.ShuffleRows(validationDataView, shufflePoolSize: 1_000_000);
            testDataView = mlContext.Data.ShuffleRows(testDataView, shufflePoolSize: 1_000_000);

            // Build training pipeline
            //IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipelineUsingAutoML(mlContext, trainingDataView, validationDataView, useAutoML: true);

            // Train Model
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            // Evaluate quality of Model
            EvaluateModel(mlContext, mlModel, testDataView);

            // Save model
            SaveModel(mlContext, mlModel, modelFileName, trainingDataView.Schema);

            // Print model weights of a proxy model
            PrintProxyModelWeights(mlContext, trainingDataView, trainingPipeline);
        }

        //public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        //{
        //    // Data process configuration with pipeline data transformations 
        //    var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
        //        // Reset the Weight column to "1.0"
        //        .Append(mlContext.Transforms.Expression("Weight", "x : 1.0f", new[] { "Weight"} ))

        //        // Numeric features
        //        .Append(mlContext.Transforms.Concatenate("FeaturesNumeric", new[] { "tagAvgFollowers", "tagSumFollowers", "tagAvgTotalItems", "tagSumTotalItems", "imagesCount" }))

        //        // Categorical features
        //        .Append(mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair("img1Type", "img1Type"), new InputOutputColumnPair("img2Type", "img2Type"), new InputOutputColumnPair("img3Type", "img3Type") }))
        //        .Append(mlContext.Transforms.Concatenate("FeaturesCategorical", new[] { "img1Type", "img2Type", "img3Type" }))

        //        // Text features
        //        .Append(mlContext.Transforms.Text.FeaturizeText("imgDesc_tf", null, new[] { "img1Desc", "img2Desc", "img3Desc" }))
        //        .Append(mlContext.Transforms.Text.FeaturizeText("tags_tf", "tags"))
        //        .Append(mlContext.Transforms.Text.FeaturizeText("title_tf", "title"))
        //        .Append(mlContext.Transforms.Concatenate("FeaturesText", new[] { "title_tf", "tags_tf", "imgDesc_tf" }))

        //        // Model stacking using an Averaged Perceptron on the text features
        //        //.Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "FeaturesText", numberOfIterations: 10), labelColumnName: "Label"))
        //        //.Append(mlContext.Transforms.Concatenate("FeaturesStackedAPOnText", new[] { "Score" })) // Score column from the stacked trainer

        //        // Image features
        //        .Append(mlContext.Transforms.LoadImages("ImageObject", basepath, "img1FileName"))
        //        .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
        //        .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
        //        .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage1", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"))
        //        /*
        //        .Append(mlContext.Transforms.LoadImages("ImageObject", basepath, "img2FileName"))
        //        .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
        //        .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
        //        .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage2", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"))

        //        .Append(mlContext.Transforms.LoadImages("ImageObject", basepath, "img3FileName"))
        //        .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
        //        .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
        //        .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage3", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"))
        //        (*/
        //        .Append(mlContext.Transforms.Concatenate("FeaturesImage", new[] { "FeaturesImage1" /*, "FeaturesImage2", "FeaturesImage3"*/ }))
        //        .AppendCacheCheckpoint(env: mlContext) // Cache checkpoint since the DnnFeaturizeImage is slow

        //        // Model stacking using a logistic regression on the image features to re-learn the output layer of the ResNet model(s) 
        //        .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "FeaturesImage"))
        //        .Append(mlContext.Transforms.Concatenate("FeaturesStackedLROnImages", new[] { "Score" })) // Score column from the stacked trainer

        //        .Append(mlContext.Transforms.Concatenate("Features", new[] {
        //            "FeaturesNumeric",
        //            "FeaturesCategorical",
        //            "FeaturesText",
        //            //"FeaturesStackedAPOnText",
        //            //"FeaturesImage",
        //            "FeaturesStackedLROnImages",
        //        }))
        //        .AppendCacheCheckpoint(mlContext);

        //    // Set the training algorithm 
        //    //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features", numberOfIterations: 10), labelColumnName: "Label")
        //    //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"), labelColumnName: "Label")
        //    var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(enforceNonNegativity: true, exampleWeightColumnName: "Weight", featureColumnName: "Features", labelColumnName: "Label")
        //                              .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
        //    var trainingPipeline = dataProcessPipeline.Append(trainer);

        //    return trainingPipeline;
        //}

        public static IEstimator<ITransformer> BuildTrainingPipelineUsingAutoML(MLContext mlContext, IDataView trainDataView, IDataView validationDataView, bool useAutoML)
        {

            ExperimentResult<MulticlassClassificationMetrics> experimentResult1 = null;
            ExperimentResult<MulticlassClassificationMetrics> experimentResult2 = null;

            if (useAutoML)
            {
                var useThreads = false;

                if (!useThreads)
                {
                    experimentResult1 = TrainAutoMLSubPipeline1(mlContext, trainDataView, validationDataView);
                    experimentResult2 = TrainAutoMLSubPipeline2(mlContext, trainDataView, validationDataView);
                }
                else
                {
                    Thread workerThread1 = new(() => experimentResult1 = TrainAutoMLSubPipeline1(mlContext, trainDataView, validationDataView));
                    workerThread1.Name = "AutoML Worker Thread - Text  ";

                    Thread workerThread2 = new(() => experimentResult2 = TrainAutoMLSubPipeline2(mlContext, trainDataView, validationDataView));
                    workerThread2.Name = "AutoML Worker Thread - Images";

                    workerThread1.Start();
                    workerThread2.Start();

                    workerThread1.Join();
                    workerThread2.Join();
                }
            }

            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline =
                // Up weight the FrontPage posts to handle the class imblanace
                mlContext.Transforms.Expression("Weight", "x : (x == \"UserSub\" ? 1f : 100.0f)", new[] { "Label" })

                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Label"))

                // Numeric features
                .Append(mlContext.Transforms.Concatenate("FeaturesNumeric", new[] { "tagAvgFollowers", "tagSumFollowers", "tagAvgTotalItems", "tagSumTotalItems", "imagesCount", "tagCount", "tagMaxFollowers", "tagMaxTotalItems" }))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance("FeaturesNumeric")) // Log scaling due to the wide scale disparity of tag followers/posts

                // Categorical features
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair("img1Type", "img1Type"), new InputOutputColumnPair("img2Type", "img2Type"), new InputOutputColumnPair("img3Type", "img3Type") }))
                .Append(mlContext.Transforms.Concatenate("FeaturesCategorical", new[] { "img1Type", "img2Type", "img3Type" }))

                // Text features
                .Append(mlContext.Transforms.Text.FeaturizeText("imgDesc_tf", null, new[] { "img1Desc", "img2Desc", "img3Desc" }))
                .Append(mlContext.Transforms.Text.FeaturizeText("tags_tf", "tags"))
                .Append(mlContext.Transforms.Text.FeaturizeText("title_tf", "title"))
                .Append(mlContext.Transforms.Concatenate("FeaturesText", new[] { "title_tf", "tags_tf", "imgDesc_tf" }))

                // Model stacking using an Averaged Perceptron on the text features
                .AppendCacheCheckpoint(env: mlContext) // Cache checkpoint since the OVA Averaged Perceptron makes many passes of its data
                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "FeaturesText", numberOfIterations: 10), labelColumnName: "Label")) // todo: file bug that AveragedPerceptron does not expose exampleWeightColumnName
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedAPOnText", new[] { "Score" })) // Score column from the stacked trainer

                // Model stacking using k-means as a featurizer on the numeric features (note: they were normalized above)
                .Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "FeaturesNumeric", exampleWeightColumnName: "Weight", numberOfClusters: 10)) // todo: file bug that the initialization method is not exposed; random init is way faster for a larger number of clusters
                .Append(mlContext.Transforms.Concatenate("FeaturesKMeansClusterDistanceOnNumeric", new[] { "Score" })) // Score column from the stacked trainer

                // String statistics (length, vowelCount, numberCount, ...) on title, img1Desc, img2Desc, img3Desc
                .Append(mlContext.Transforms.CopyColumns("text", "title"))
                .Append(mlContext.Transforms.CustomMapping(new StringStatisticsAction().GetMapping(), "StringStatistics"))
                .Append(mlContext.Transforms.Concatenate("StringStatsOnTitle", new[] { "length", "vowelCount", "consonantCount", "numberCount", "underscoreCount", "letterCount", "wordCount", "wordLengthAverage", "lineCount", "startsWithVowel", "endsInVowel", "endsInVowelNumber", "lowerCaseCount", "upperCaseCount", "upperCasePercent", "letterPercent", "numberPercent", "longestRepeatingChar", "longestRepeatingVowel" }))
                .Append(mlContext.Transforms.CopyColumns("text", "img1Desc"))
                .Append(mlContext.Transforms.CustomMapping(new StringStatisticsAction().GetMapping(), "StringStatistics"))
                .Append(mlContext.Transforms.Concatenate("StringStatsOnImg1Desc", new[] { "length", "vowelCount", "consonantCount", "numberCount", "underscoreCount", "letterCount", "wordCount", "wordLengthAverage", "lineCount", "startsWithVowel", "endsInVowel", "endsInVowelNumber", "lowerCaseCount", "upperCaseCount", "upperCasePercent", "letterPercent", "numberPercent", "longestRepeatingChar", "longestRepeatingVowel" }))
                .Append(mlContext.Transforms.CopyColumns("text", "img2Desc"))
                .Append(mlContext.Transforms.CustomMapping(new StringStatisticsAction().GetMapping(), "StringStatistics"))
                .Append(mlContext.Transforms.Concatenate("StringStatsOnImg2Desc", new[] { "length", "vowelCount", "consonantCount", "numberCount", "underscoreCount", "letterCount", "wordCount", "wordLengthAverage", "lineCount", "startsWithVowel", "endsInVowel", "endsInVowelNumber", "lowerCaseCount", "upperCaseCount", "upperCasePercent", "letterPercent", "numberPercent", "longestRepeatingChar", "longestRepeatingVowel" }))
                .Append(mlContext.Transforms.CopyColumns("text", "img3Desc"))
                .Append(mlContext.Transforms.CustomMapping(new StringStatisticsAction().GetMapping(), "StringStatistics"))
                .Append(mlContext.Transforms.Concatenate("StringStatsOnImg3Desc", new[] { "length", "vowelCount", "consonantCount", "numberCount", "underscoreCount", "letterCount", "wordCount", "wordLengthAverage", "lineCount", "startsWithVowel", "endsInVowel", "endsInVowelNumber", "lowerCaseCount", "upperCaseCount", "upperCasePercent", "letterPercent", "numberPercent", "longestRepeatingChar", "longestRepeatingVowel" }))
                .Append(mlContext.Transforms.Concatenate("FeaturesStringStatistics", new[] { "StringStatsOnTitle", "StringStatsOnImg1Desc", "StringStatsOnImg2Desc", "StringStatsOnImg3Desc" }))

                // Model stacking using k-means as a featurizer on the string statistics features
                .Append(mlContext.Transforms.NormalizeMinMax("FeaturesStringStatisticsNorm", "FeaturesStringStatistics"))
                .Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "FeaturesStringStatisticsNorm", exampleWeightColumnName: "Weight", numberOfClusters: 10))
                .Append(mlContext.Transforms.Concatenate("FeaturesKMeansClusterDistanceOnStringStats", new[] { "Score" })) // Score column from the stacked trainer
                
                // Model stacking using a LightGBM on the string statistics
                .Append(mlContext.MulticlassClassification.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "FeaturesStringStatistics", exampleWeightColumnName: "Weight"))
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedLGBMOnStringStats", new[] { "Score" })) // Score column from the stacked trainer

                // Model stacking using an FastTreeTweedie on text/categorical/numeric features towards the PostScore alt-label (predicts PostScore without leaking)
                .Append(mlContext.Transforms.Concatenate("FeaturesForStackedModel", new[] { "FeaturesNumeric", "FeaturesCategorical", "FeaturesText", }))
                .Append(mlContext.Transforms.Expression("postScoreLog", "x : log(max(x,-1) + 2)", new[] { "postScore" })) // Log of the postScore, since the postScore scales multiple decades fo range
                .Append(mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "postScoreLog", featureColumnName: "FeaturesForStackedModel", exampleWeightColumnName: "Weight"))
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedFastTreeTweedieToPostScoreLog", new[] { "Score" })) // Score column from the stacked trainer

                // Image features
                .Append(mlContext.Transforms.LoadImages("ImageObject", basepath, "img1FileName"))
                .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
                .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
                .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage1", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels")) // todo: file bug that Microsoft.ML.OnnxRuntime & (maybe) Microsoft.ML.OnnxRuntime.Managed nugets need to be listed as a dependency of Microsoft.ML.DnnImageFeaturizer.*. // todo: file bug that ResNet50/101 does not work. 
                .Append(mlContext.Transforms.Concatenate("FeaturesImage", new[] { "FeaturesImage1" }))
                .AppendCacheCheckpoint(env: mlContext) // Cache checkpoint since the DnnFeaturizeImage is slow

                // Model stacking using a logistic regression on the image features to re-learn the output layer of the ResNet model 
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "FeaturesImage", exampleWeightColumnName: "Weight"))
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedLROnImages", new[] { "Score" })); // Score column from the stacked trainer

            // Append the AutoML stacked models if enabled
            if (useAutoML)
            {
                dataProcessPipeline = dataProcessPipeline // New variable would be needed if the last transform wasn't also a Concatenate, due to typing issues

                    // Model stacking using the previously learned AutoML model #1 on text, categorical, and numeric features (note: the AutoML estimator chain may mask existing columns)
                    .Append(experimentResult1.BestRun.Estimator)
                    .Append(mlContext.Transforms.Concatenate("FeaturesStackedAutoMLOnText", new[] { "Score" })) // Score column from the stacked trainer

                    // Model stacking using the previously learned AutoML model #2 on Imge features (note: the AutoML estimator chain may mask existing columns)
                    .Append(experimentResult2.BestRun.Estimator)
                    .Append(mlContext.Transforms.Concatenate("FeaturesStackedAutoMLOnImages", new[] { "Score" })); // Score column from the stacked trainer

                var automlFeatureColumns = new[] { "FeaturesStackedAutoMLOnTextCatNum", "FeaturesStackedAutoMLOnImages" };
            }

            dataProcessPipeline = dataProcessPipeline
                .Append(mlContext.Transforms.Concatenate("Features",
                    new[] {
                        "FeaturesNumeric",
                        "FeaturesCategorical",
                        "FeaturesText",
                        "FeaturesStackedAPOnText",
                        //"FeaturesImage",
                        "FeaturesStackedLROnImages",
                        "FeaturesStackedFastTreeTweedieToPostScoreLog",
                        "FeaturesStringStatistics",
                        "FeaturesStackedLGBMOnStringStats",
                        "FeaturesKMeansClusterDistanceOnNumeric",
                        "FeaturesKMeansClusterDistanceOnStringStats",
                    }
                    // Concatenate the AutoML features if enabled
                    .Concat(useAutoML ? new[] { "FeaturesStackedAutoMLOnText", "FeaturesStackedAutoMLOnImages" } : new string[] { }).ToArray()))
                .AppendCacheCheckpoint(mlContext);


            // Set the training algorithm 
            var trainer =
                //mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features", numberOfIterations: 10), labelColumnName: "Label")
                mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features", exampleWeightColumnName: "Weight", minimumExampleCountPerLeaf: 2), labelColumnName: "Label")
                    //mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Label", featureColumnName: "Features", exampleWeightColumnName: "Weight"), labelColumnName: "Label")
                    //mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(enforceNonNegativity: true, exampleWeightColumnName: "Weight", featureColumnName: "Features", labelColumnName: "Label")
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;

        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            var sw = new Stopwatch();
            sw.Start();

            Console.WriteLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine($"=============== End of training process ({sw.ElapsedMilliseconds / 1000.0} sec) ===============\n");
            return model;
        }


        private static void EvaluateModel(MLContext mlContext, ITransformer mlModel, IDataView testDataView)
        {
            var sw = new Stopwatch();
            sw.Start();

            // Evaluate the model and show accuracy stats
            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            IDataView predictions = mlModel.Transform(testDataView);

            // Obtuse method of getting the number of classes
            VBuffer<ReadOnlyMemory<char>> classNamesVBuffer = default;
            predictions.Schema["Score"].GetSlotNames(ref classNamesVBuffer);
            var numClasses = classNamesVBuffer.Length;
            string[] classNames = classNamesVBuffer.DenseValues().Select(a => a.ToString()).ToArray();

            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score", topKPredictionCount: numClasses); // Todo: fix bug to allow for `topKPredictionCount: Int32.MaxValue` 
            ConsoleHelper.PrintMulticlassClassificationMetrics(metrics, classNames);
            Console.WriteLine($"===== Finished Evaluating Model's accuracy with Test data ({sw.ElapsedMilliseconds / 1000.0} sec) =====");
        }


        private static ExperimentResult<MulticlassClassificationMetrics> TrainAutoMLSubPipeline1(MLContext mlContext, IDataView trainDataView, IDataView validationDataView)
        {
            var sw = new Stopwatch();
            sw.Start();

            ExperimentResult<MulticlassClassificationMetrics> experimentResult;

            //Console.WriteLine("\n=============== Training Stacked AutoML model on Text, Categorical, Numeric features ===============");
            Console.WriteLine("\n=============== Training Stacked AutoML model on Text features ===============");

            Func<MulticlassClassificationMetrics> GetBaselineMetrics = () =>
            {
                var baselinePipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                    .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.Prior(labelColumnName: "Label", exampleWeightColumnName: "Weight"), labelColumnName: "Label"));
                var model = baselinePipeline.Fit(trainDataView);
                var predictions = model.Transform(validationDataView);
                var baselineMetrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");
                return baselineMetrics;
            };

            var progressHandler = new MulticlassExperimentProgressHandler(GetBaselineMetrics);

            var experimentSettings = new MulticlassExperimentSettings
            {
                MaxExperimentTimeInSeconds = 3600,
                OptimizingMetric = MulticlassClassificationMetric.LogLossReduction,
                //OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
            };

            // Set the column purposes for a subset of columns; the rest are auto-inferred
            ColumnInformation columnInformation = new ColumnInformation();
            columnInformation.ExampleWeightColumnName = "Weight";
            columnInformation.LabelColumnName = "Label";
            var textColumns = new List<string>() { "tags", "img1Desc", "img2Desc", "img3Desc" }; // This correct the misprediction of the column purpose. These are otherwise incorrectly infered as categorical-hash due to sparsity. todo: fix https://github.com/dotnet/machinelearning/issues/3879
            textColumns.ForEach(a => columnInformation.TextColumnNames.Add(a));

            var columnsToIgnore = new List<string>() {
                // Non-useful columns
                "Name", "img1Uri", "img2Uri",  "img3Uri", // Why to ignore: urls contain no information

                // Leaky columns
                "img1FileName",  "img2FileName", "img3FileName", // Why to ignore: filenames leak the label since the label is in the path
                "datetime", // Why to ignore: datetime leaks the label due to the collection method

                // Alt-labels (also leaky)
                "views", "ups", "downs", "points", "postScore", "commentCount", "favoriteCount", // Why to ignore: alt-labels (other labels that we could train towards)

                // Categorical features
                "img1Type", "img2Type", "img3Type", // Why to ignore: model will focus on text, so we'll ignore the categorical columns

                // Numeric features
                "tagAvgFollowers", "tagSumFollowers", "tagAvgTotalItems", "tagSumTotalItems", "imagesCount", "tagCount", "tagMaxFollowers", "tagMaxTotalItems", // Why to ignore: model will focus on text, so we'll ignore the numeric columns
            }; 

            columnsToIgnore.ForEach(a => columnInformation.IgnoredColumnNames.Add(a));

            experimentResult = mlContext.Auto()
                .CreateMulticlassClassificationExperiment(experimentSettings)
                .Execute(
                    trainData: trainDataView,
                    validationData: validationDataView,
                    progressHandler: progressHandler,
                    columnInformation: columnInformation);

            Console.WriteLine("\nBest run:");
            //progressHandler.Report(experimentResult.BestRun);
            ConsoleHelper.PrintIterationMetrics(experimentResult.RunDetails.ToList().IndexOf(experimentResult.BestRun),
                experimentResult.BestRun.TrainerName, experimentResult.BestRun.ValidationMetrics, experimentResult.BestRun.RuntimeInSeconds);
            Console.WriteLine($"=============== Finished training AutoML model ({sw.ElapsedMilliseconds / 1000.0} sec) ===============");

            return experimentResult;
        }


        private static ExperimentResult<MulticlassClassificationMetrics> TrainAutoMLSubPipeline2(MLContext mlContext, IDataView trainDataView, IDataView validationDataView)
        {
            var sw = new Stopwatch();
            sw.Start();

            ExperimentResult<MulticlassClassificationMetrics> experimentResult;

            Console.WriteLine("\n=============== Training Stacked AutoML model on Image features ===============");

            Func<MulticlassClassificationMetrics> GetBaselineMetrics = () =>
            {
                var baselinePipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                    .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.Prior(labelColumnName: "Label", exampleWeightColumnName: "Weight"), labelColumnName: "Label"));
                var model = baselinePipeline.Fit(trainDataView);
                var predictions = model.Transform(validationDataView);
                var baselineMetrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");
                return baselineMetrics;
            };

            var progressHandler = new MulticlassExperimentProgressHandler(GetBaselineMetrics);

            var experimentSettings = new MulticlassExperimentSettings
            {
                MaxExperimentTimeInSeconds = 40 * 60,
                OptimizingMetric = MulticlassClassificationMetric.LogLossReduction,
                //OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
            };

            // Set the column purposes for a subset of columns; the rest are auto-inferred
            ColumnInformation columnInformation = new ColumnInformation();
            columnInformation.ExampleWeightColumnName = "Weight";
            columnInformation.LabelColumnName = "Label";


            var columnsToIgnore = new List<string>() {
                // Non-useful columns
                "Name", "img1Uri", "img2Uri",  "img3Uri", // Why to ignore: urls contain no information

                // Leaky columns
                "img1FileName",  "img2FileName", "img3FileName", // Why to ignore: filenames leak the label since the label is in the path
                "datetime", // Why to ignore: datetime leaks the label due to the collection method

                // Alt-labels (also leaky)
                "views", "ups", "downs", "points", "postScore", "commentCount", "favoriteCount", // Why to ignore: alt-labels (other labels that we could train towards)

                // Categorical features
                "img1Type", "img2Type", "img3Type", // Why to ignore: model will focus on images, so we'll ignore the categorical columns

                // Numeric features
                "tagAvgFollowers", "tagSumFollowers", "tagAvgTotalItems", "tagSumTotalItems", "imagesCount", "tagCount", "tagMaxFollowers", "tagMaxTotalItems", // Why to ignore: model will focus on images, so we'll ignore the numeric columns

                // Text features
                 "title", "tags", "img1Desc", "img2Desc", "img3Desc",  // Why to ignore: model will focus on images, so we'll ignore the text columns

                //"ImageObject", "Pixels", "FeaturesImage1" // Why to ignore: these are intermediate columns added from the image preFeaturizer
                // todo: file bug to have AutoML check for columns to set the purpose of after the preFeaturiizer runs. Can't ignore them.
            };


            columnsToIgnore.ForEach(a => columnInformation.IgnoredColumnNames.Add(a));

            var preFeaturizer =
                mlContext.Transforms.LoadImages("ImageObject", basepath, "img1FileName")
                .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
                .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
                .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage1", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"))
                /*
                .Append(mlContext.Transforms.LoadImages("ImageObject", basepath, "img2FileName"))
                .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
                .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
                .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage2", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"))
                
                .Append(mlContext.Transforms.LoadImages("ImageObject", basepath, "img3FileName"))
                .Append(mlContext.Transforms.ResizeImages("ImageObject", imageWidth: 224, imageHeight: 224))
                .Append(mlContext.Transforms.ExtractPixels("Pixels", "ImageObject"))
                .Append(mlContext.Transforms.DnnFeaturizeImage("FeaturesImage3", m => m.ModelSelector.ResNet18(mlContext, m.OutputColumn, m.InputColumn), "Pixels"))
                */
                .Append(mlContext.Transforms.Concatenate("FeaturesImage", new[] { "FeaturesImage1" /*, "FeaturesImage2", "FeaturesImage3" */ }))
                .Append(mlContext.Transforms.DropColumns(new[] {
                    "ImageObject", "Pixels", "FeaturesImage1",  // Drop intermediate featues so AutoML won't read them and be slowed down
                    "OriginalInput", "PreprocessedInput", "Input247", "Pooling395_Output_0", // Drop features created from the ONNX model backing DnnFeaturizeImage (todo: file bug so these are automatically dropped)
                }))
                .AppendCacheCheckpoint(env: mlContext); // Cache checkpoint since the DnnFeaturizeImage is slow

            var fitPrefeaturizer = preFeaturizer.Fit(trainDataView);

            //trainDataView = mlContext.Data.ShuffleRows(input: trainDataView, shufflePoolSize: 1_000_000);
            //trainDataView = mlContext.Data.TakeRows(trainDataView, 50);

            trainDataView = fitPrefeaturizer.Transform(trainDataView);
            validationDataView = fitPrefeaturizer.Transform(validationDataView);

            trainDataView = mlContext.Data.Cache(trainDataView, new[] { "Label", "Weight", "FeaturesImage" }); // Cache checkpoint since the DnnFeaturizeImage is slow
            validationDataView = mlContext.Data.Cache(validationDataView, new[] { "Label", "Weight", "FeaturesImage" }); // Cache checkpoint since the DnnFeaturizeImage is slow

            experimentResult = mlContext.Auto()
                .CreateMulticlassClassificationExperiment(experimentSettings)
                .Execute(
                    //trainData: mlContext.Data.TakeRows(mlContext.Data.ShuffleRows(input: trainDataView, shufflePoolSize: 5000), 100),
                    trainData: trainDataView,
                    validationData: validationDataView,
                    progressHandler: progressHandler,
                    columnInformation: columnInformation //,
                                                         //preFeaturizer: preFeaturizer
                  );

            Console.WriteLine("\nBest run:");
            //progressHandler.Report(experimentResult.BestRun);
            ConsoleHelper.PrintIterationMetrics(experimentResult.RunDetails.ToList().IndexOf(experimentResult.BestRun),
                experimentResult.BestRun.TrainerName, experimentResult.BestRun.ValidationMetrics, experimentResult.BestRun.RuntimeInSeconds);
            Console.WriteLine($"=============== Finished training AutoML model ({sw.ElapsedMilliseconds / 1000.0} sec) ===============");

            return experimentResult;
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
                ExampleWeightColumnName = "Weight",

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

            var sw = new Stopwatch();
            sw.Start();
            Console.WriteLine("=============== Training proxy model ===============");

            // Fit this pipeline to the training data.
            var model = pipeline.Fit(trainingDataView);

            Console.WriteLine($"=============== End of proxy training process ({sw.ElapsedMilliseconds / 1000.0} sec) ===============\n");

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
                .Select((x, i) => ($"{(x.Value.Length > 0 ? x.Value : $"UnnamedSlot_{i:000000}")}", weightsValues[i], (float)Math.Abs(weightsValues[i])))
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

        #region StringStatistics CustomMapping
        [CustomMappingFactoryAttribute("StringStatistics")]
        private class StringStatisticsAction : CustomMappingFactory<RowWithText, RowWithStringStatistics>
        {
            
            public static Action<RowWithText, RowWithStringStatistics> CustomAction = (RowWithText input, RowWithStringStatistics output) =>
            {
                string str = (input.Text is string ? (string)(object)input.Text : string.Join(" ", input.Text));
                char[] text = str.ToCharArray();

                // Note: These are written for clarity; for speed, a single pass of the character array could be done.
                output.Length = text.Length;
                output.VowelCount = text.Count(isVowel);
                output.ConsonantCount = text.Count(isConsonant);
                output.NumberCount = text.Count(Char.IsDigit);
                output.UnderscoreCount = text.Count(c => c == '_');
                output.LetterCount = text.Count(Char.IsLetter);
                output.WordCount = text.Count(Char.IsSeparator) + 1;
                output.WordLengthAverage = (output.Length - output.WordCount + 1) / output.WordCount;
                output.LineCount = text.Count(c => c == '\n') + 1;
                output.StartsWithVowel = (isVowel(text.FirstOrDefault()) ? 1 : 0);
                output.EndsInVowel = (isVowel(text.LastOrDefault()) ? 1 : 0);
                output.EndsInVowelNumber = (isVowelOrDigit(text.LastOrDefault()) ? 1 : 0);
                output.LowerCaseCount = text.Count(Char.IsLower);
                output.UpperCaseCount = text.Count(Char.IsUpper);
                output.UpperCasePercent = (output.LetterCount == 0 ? 0 : ((float)output.UpperCaseCount) / output.LetterCount);
                output.LetterPercent = (output.Length == 0 ? 0 : ((float)output.LetterCount) / output.Length);
                output.NumberPercent = (output.Length == 0 ? 0 : ((float)output.NumberCount) / output.Length);
                output.LongestRepeatingChar = maxRepeatingCharCount(text);
                output.LongestRepeatingVowel = maxRepeatingVowelCount(text);
            };

            private static readonly Func<char, bool> isVowel = ((x) => x == 'e' || x == 'a' || x == 'o' || x == 'i' || x == 'u' || x == 'E' || x == 'A' || x == 'O' || x == 'I' || x == 'U');
            private static readonly Func<char, bool> isConsonant = ((x) => (x >= 'A' && x <= 'Z') || (x >= 'a' && x <= 'z') && !(x == 'e' || x == 'a' || x == 'o' || x == 'i' || x == 'u' || x == 'E' || x == 'A' || x == 'O' || x == 'I' || x == 'U'));
            private static readonly Func<char, bool> isVowelOrDigit = ((x) => x == 'e' || x == 'a' || x == 'o' || x == 'i' || x == 'u' || x == 'E' || x == 'A' || x == 'O' || x == 'I' || x == 'U' || (x >= '0' && x <= '9'));
            private static readonly Func<char[], int> maxRepeatingCharCount = ((s) => { int max = 0, j = 0; for (var i = 0; i < s.Length; ++i) { if (s[i] == s[j]) { if (max < i - j + 1) max = i - j + 1; } else j = i; } return max; });
            private static readonly Func<char[], int> maxRepeatingVowelCount = ((s) => { int max = 0, j = 0; for (var i = 0; i < s.Length; ++i) { if (s[i] == s[j] && isVowel(s[j])) { if (max < i - j + 1) max = i - j + 1; } else j = i; } return max; });

            public override Action<RowWithText, RowWithStringStatistics> GetMapping() => CustomAction;
        }

        private class RowWithText
        {
            [ColumnName("text")]
            public string Text { get; set; }
        }

        private class RowWithStringStatistics
        {
            [ColumnName("length")]
            public float Length { get; set; }

            [ColumnName("vowelCount")]
            public float VowelCount { get; set; }

            [ColumnName("consonantCount")]
            public float ConsonantCount { get; set; }

            [ColumnName("numberCount")]
            public float NumberCount { get; set; }

            [ColumnName("underscoreCount")]
            public float UnderscoreCount { get; set; }

            [ColumnName("letterCount")]
            public float LetterCount { get; set; }

            [ColumnName("wordCount")]
            public float WordCount { get; set; }

            [ColumnName("wordLengthAverage")]
            public float WordLengthAverage { get; set; }

            [ColumnName("lineCount")]
            public float LineCount { get; set; }

            [ColumnName("startsWithVowel")]
            public float StartsWithVowel { get; set; }

            [ColumnName("endsInVowel")]
            public float EndsInVowel { get; set; }

            [ColumnName("endsInVowelNumber")]
            public float EndsInVowelNumber { get; set; }

            [ColumnName("lowerCaseCount")]
            public float LowerCaseCount { get; set; }

            [ColumnName("upperCaseCount")]
            public float UpperCaseCount { get; set; }

            [ColumnName("upperCasePercent")]
            public float UpperCasePercent { get; set; }

            [ColumnName("letterPercent")]
            public float LetterPercent { get; set; }

            [ColumnName("numberPercent")]
            public float NumberPercent { get; set; }

            [ColumnName("longestRepeatingChar")]
            public float LongestRepeatingChar { get; set; }

            [ColumnName("longestRepeatingVowel")]
            public float LongestRepeatingVowel { get; set; }
        }
        #endregion

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
        private readonly Func<MulticlassClassificationMetrics> GetBaselineMetrics;

        public MulticlassExperimentProgressHandler(Func<MulticlassClassificationMetrics> getBaselineMetrics)
        {
            GetBaselineMetrics = getBaselineMetrics;
        }

        public void Report(RunDetail<MulticlassClassificationMetrics> iterationResult)
        {
            if (_iterationIndex++ == 0)
            {
                ConsoleHelper.PrintMulticlassClassificationMetricsHeader();

                // todo: Disabled due to bug in OVA-PriorPredictor. See bug: https://github.com/dotnet/machinelearning/issues/5575
                // Print baseline metrics
                //var sw = new Stopwatch();
                //sw.Start();
                //var baselineMetrics = GetBaselineMetrics();
                //var runTime = sw.ElapsedMilliseconds / 1000.0;
                //ConsoleHelper.PrintIterationMetrics(-1, "Baseline", baselineMetrics, runTime);
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
            Console.WriteLine(Thread.CurrentThread.Name + ": |" + message.PadRight(width - 2) + "|");
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
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt(values.Count() - 1);
            return confidenceInterval95;
        }
    }
}
