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
using Microsoft.ML.Transforms.Text;
using ImgurClassifier.Extras;

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

        public static void CreateModel(MLContext mlContext, Action<string> writeLogLine)
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
            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipelineUsing(mlContext, trainingDataView, validationDataView, useAutoML: false, writeLogLine);

            // Train Model
            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline, writeLogLine);

            // Evaluate quality of Model
            EvaluateModel(mlContext, mlModel, testDataView, writeLogLine);

            // Save model
            SaveModel(mlContext, mlModel, modelFileName, trainingDataView.Schema, writeLogLine);

            // Print model weights of a proxy model
            Utils.ProxyModelFeatureImportance(mlContext, Utils.Task.MulticlassClassification, "Label", "Features", "Weight", trainingDataView, trainingPipeline, writeLogLine);
        }

        
        public static IEstimator<ITransformer> BuildTrainingPipelineUsing(MLContext mlContext, IDataView trainDataView, IDataView validationDataView, bool useAutoML, Action<string> writeLogLine)
        {

            ExperimentResult<MulticlassClassificationMetrics> experimentResult1 = null;
            ExperimentResult<MulticlassClassificationMetrics> experimentResult2 = null;
            ExperimentResult<MulticlassClassificationMetrics> experimentResult3 = null;

            if (useAutoML)
            {
                var useThreads = false;

                if (!useThreads)
                {
                    experimentResult2 = TrainAutoMLSubPipeline2(mlContext, trainDataView, validationDataView, writeLogLine);
                    experimentResult3 = TrainAutoMLSubPipeline3(mlContext, trainDataView, validationDataView, writeLogLine);
                    experimentResult1 = TrainAutoMLSubPipeline1(mlContext, trainDataView, validationDataView, writeLogLine);
                }
                else
                {
                    Thread workerThread1 = new(() => experimentResult1 = TrainAutoMLSubPipeline1(mlContext, trainDataView, validationDataView, writeLogLine));
                    workerThread1.Name = "AutoML Worker Thread - Text  ";

                    Thread workerThread2 = new(() => experimentResult2 = TrainAutoMLSubPipeline2(mlContext, trainDataView, validationDataView, writeLogLine));
                    workerThread2.Name = "AutoML Worker Thread - Images";

                    Thread workerThread3 = new(() => experimentResult3 = TrainAutoMLSubPipeline3(mlContext, trainDataView, validationDataView, writeLogLine));
                    workerThread3.Name = "AutoML Worker Thread - Images using TensorFlow";

                    workerThread1.Start();
                    workerThread2.Start();
                    workerThread3.Start();

                    workerThread1.Join();
                    workerThread2.Join();
                    workerThread3.Join();
                }
            }

            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline =
                // Up weight the FrontPage posts to handle the class imblanace
                mlContext.Transforms.Expression("Weight", "x : (x == \"UserSub\" ? 1.0f : 100.0f)", new[] { "Label" })

                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Label", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)) // ByValue makes the class ordering stable, making it easier to compare metrics between runs

                // Numeric features
                .Append(mlContext.Transforms.Concatenate("FeaturesNumeric", new[] { "tagAvgFollowers", "tagSumFollowers", "tagAvgTotalItems", "tagSumTotalItems", "imagesCount", "tagCount", "tagMaxFollowers", "tagMaxTotalItems" }))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance("FeaturesNumeric")) // Log scaling due to the wide scale disparity of tag followers/posts

                // Categorical features
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair("img1Type", "img1Type"), new InputOutputColumnPair("img2Type", "img2Type"), new InputOutputColumnPair("img3Type", "img3Type") }))
                .Append(mlContext.Transforms.Concatenate("FeaturesCategorical", new[] { "img1Type", "img2Type", "img3Type" }))

                 // Text features (bigrams + trichargrams)
                 .Append(mlContext.Transforms.Text.FeaturizeText("FeaturesText", new TextFeaturizingEstimator.Options() { OutputTokensColumnName = "TokensForWordEmbedding" }, new[] { "title", "img1Desc", "img2Desc", "img3Desc" })) // TokensForWordEmbedding is later used by the word embedding transform

                 // Text features (bigrams + trichargrams) on just the tags
                 .Append(mlContext.Transforms.Text.FeaturizeText("FeaturesTextOnTags", new TextFeaturizingEstimator.Options() { }, new[] { "tags" }))

                // Count Target Encoder (should run on tag names instead)
                //.Append(mlContext.Transforms.CountTargetEncode("FeaturesCountTargetEncoder", "img1Type")) // todo: Log bug as CountTargetEncode seems to be only partially exposed in the estimator API

                // Model stacking using an Averaged Perceptron on the text features
                .AppendCacheCheckpoint(env: mlContext) // Cache checkpoint since the OVA Averaged Perceptron makes many passes of its data
                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "FeaturesText", numberOfIterations: 10), labelColumnName: "Label")) // todo: file bug that AveragedPerceptron does not expose exampleWeightColumnName
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedAPOnText", new[] { "Score" })) // Score column from the stacked trainer

                // Model stacking using k-means as a featurizer on the numeric features (note: they were normalized above)
                .Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "FeaturesNumeric", exampleWeightColumnName: "Weight", numberOfClusters: 10)) // todo: file bug that the initialization method is not exposed; random init is way faster for a larger number of clusters
                .Append(mlContext.Transforms.Concatenate("FeaturesKMeansClusterDistanceOnNumeric", new[] { "Score" })) // Score column from the stacked trainer

                // String statistics (length, vowelCount, numberCount, ...) on title, img1Desc, img2Desc, img3Desc
                .Append(mlContext.Transforms.CopyColumns("text", "title"))
                .Append(mlContext.Transforms.CustomMapping(new StringStatisticsFeaturizer.StringStatisticsAction().GetMapping(), "StringStatistics"))
                .Append(mlContext.Transforms.Concatenate("StringStatsOnTitle", new[] { "length", "vowelCount", "consonantCount", "numberCount", "underscoreCount", "letterCount", "wordCount", "wordLengthAverage", "lineCount", "startsWithVowel", "endsInVowel", "endsInVowelNumber", "lowerCaseCount", "upperCaseCount", "upperCasePercent", "letterPercent", "numberPercent", "longestRepeatingChar", "longestRepeatingVowel" }))
                .Append(mlContext.Transforms.CopyColumns("text", "img1Desc"))
                .Append(mlContext.Transforms.CustomMapping(new StringStatisticsFeaturizer.StringStatisticsAction().GetMapping(), "StringStatistics"))
                .Append(mlContext.Transforms.Concatenate("StringStatsOnImg1Desc", new[] { "length", "vowelCount", "consonantCount", "numberCount", "underscoreCount", "letterCount", "wordCount", "wordLengthAverage", "lineCount", "startsWithVowel", "endsInVowel", "endsInVowelNumber", "lowerCaseCount", "upperCaseCount", "upperCasePercent", "letterPercent", "numberPercent", "longestRepeatingChar", "longestRepeatingVowel" }))
                .Append(mlContext.Transforms.CopyColumns("text", "img2Desc"))
                .Append(mlContext.Transforms.CustomMapping(new StringStatisticsFeaturizer.StringStatisticsAction().GetMapping(), "StringStatistics"))
                .Append(mlContext.Transforms.Concatenate("StringStatsOnImg2Desc", new[] { "length", "vowelCount", "consonantCount", "numberCount", "underscoreCount", "letterCount", "wordCount", "wordLengthAverage", "lineCount", "startsWithVowel", "endsInVowel", "endsInVowelNumber", "lowerCaseCount", "upperCaseCount", "upperCasePercent", "letterPercent", "numberPercent", "longestRepeatingChar", "longestRepeatingVowel" }))
                .Append(mlContext.Transforms.CopyColumns("text", "img3Desc"))
                .Append(mlContext.Transforms.CustomMapping(new StringStatisticsFeaturizer.StringStatisticsAction().GetMapping(), "StringStatistics"))
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

                // Tree Featurizer using a random forest regression model // todo: figure out why the tree feat isn't working here
                //.Append(mlContext.Transforms.Concatenate("FeaturesForTreeFeat", new[] { "FeaturesNumeric", "FeaturesCategorical", "FeaturesText", }))
                //.Append(mlContext.Transforms.CustomMapping(new ConvertLabelKeyToFloat().GetMapping(), "ConvertLabelKeyToFloat")) // Convert the Key type Label to a Float (so we can use a regression trainer on multiclass)
                //.Append(mlContext.Transforms.FeaturizeByFastForestRegression(options: new FastForestRegressionFeaturizationEstimator.Options { InputColumnName = "FeaturesForTreeFeat", TreesColumnName = "FeaturesTreeFeatTrees", LeavesColumnName = "FeaturesTreeFeatLeaves", PathsColumnName = "FeaturesTreeFeatPaths", TrainerOptions = new FastForestRegressionTrainer.Options { LabelColumnName = "FloatLabel", FeatureColumnName = "FeaturesForTreeFeat", ExampleWeightColumnName = "Weight", ShuffleLabels = true, /* Shuffle the label ordering before each tree is learned. Needed when running a multi-class dataset as regression. */ } }))
                //.Append(mlContext.Transforms.Concatenate("FeaturesTreeFeat", new[] { "FeaturesTreeFeatLeaves" })) // Leaves column from the stacked tree featurizer model

                // Model stacking using FastForestRegression on text/categorical/numeric features
                .Append(mlContext.Transforms.Concatenate("FeaturesForStackedModel", new[] { "FeaturesNumeric", "FeaturesCategorical", "FeaturesText", }))
                .Append(mlContext.Transforms.CustomMapping(new Utils.ConvertLabelKeyToFloat().GetMapping(), "ConvertLabelKeyToFloat")) // Convert the Key type Label to a Float (so we can use a regression trainer on multiclass)
                .Append(mlContext.Regression.Trainers.FastForest(options: new FastForestRegressionTrainer.Options
                {
                    //FeatureFirstUsePenalty = 0.1,
                    NumberOfLeaves = 20,
                    FeatureFraction = 0.7,
                    NumberOfTrees = 500,
                    LabelColumnName = "FloatLabel",
                    FeatureColumnName = "FeaturesForStackedModel",
                    ExampleWeightColumnName = "Weight",
                    //ExecutionTime = true,

                    // Shuffle the label ordering before each tree is learned.
                    // Needed when running a multi-class dataset as regression.
                    ShuffleLabels = true, // todo: remove FastForestRegression stacked model; is only for testing why the FeaturizeByFastForestRegression is failing; won't be useful w/ `ShuffleLabels = true` set
                }))
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedFastForest", new[] { "Score" })) // Score column from the stacked trainer

                // Word Embeddings
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("FeaturesWordEmbedding", "TokensForWordEmbedding", WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D)) // todo: file bug that the word embedding transform doesn't add slot names. Perhaps just slot000...slot900? Or have it default to that in the concat for unnamed slot?
                .Append(mlContext.Transforms.NormalizeMinMax("FeaturesWordEmbedding", "FeaturesWordEmbedding"))

                // Model stacking using an AveragedPerceptron on the word embedding features
                .AppendCacheCheckpoint(env: mlContext) // Cache checkpoint since the OVA Averaged Perceptron makes many passes of its data
                .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "FeaturesWordEmbedding", numberOfIterations: 10), labelColumnName: "Label"))
                .Append(mlContext.Transforms.Concatenate("FeaturesStackedAPOnWordEmbeddings", new[] { "Score" })) // Score column from the stacked trainer

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
                    .Append(mlContext.Transforms.Concatenate("FeaturesStackedAutoMLOnImages", new[] { "Score" })) // Score column from the stacked trainer
                    //.AppendCacheCheckpoint(env: mlContext) // Cache checkpoint since the DnnFeaturizeImage is slow

                    // Model stacking using the previously learned AutoML model #2 on Imge features (note: the AutoML estimator chain may mask existing columns)
                    .Append(experimentResult3.BestRun.Estimator)
                    .Append(mlContext.Transforms.Concatenate("FeaturesStackedAutoMLOnTensorFlowImages", new[] { "Score" })) // Score column from the stacked trainer
                    .AppendCacheCheckpoint(env: mlContext); // Cache checkpoint since TF is slow
            }

            string[] availableColumns = new[] {
                    "FeaturesNumeric",
                    "FeaturesCategorical",
                    "FeaturesText",
                    "FeaturesTextOnTags",
                    "FeaturesStackedAPOnText",
                    "FeaturesImage",
                    //"FeaturesTreeFeat",
                    //"FeaturesCountTargetEncoder",
                    "FeaturesStackedFastForest",
                    "FeaturesWordEmbedding",
                    "FeaturesStackedLROnImages",
                    "FeaturesStackedFastTreeTweedieToPostScoreLog",
                    "FeaturesStringStatistics",
                    "FeaturesStackedLGBMOnStringStats",
                    "FeaturesStackedAPOnWordEmbeddings",
                    "FeaturesKMeansClusterDistanceOnNumeric",
                    "FeaturesKMeansClusterDistanceOnStringStats",
                }
                // Concatenate the AutoML features if enabled
                .Concat(useAutoML ? new[] { "FeaturesStackedAutoMLOnText", "FeaturesStackedAutoMLOnImages", "FeaturesStackedAutoMLOnTensorFlowImages" } : new string[] { }).ToArray();

            // Run auto-column feature selection to choose the best columns
            var columnsToUse = Utils.AutoColumnSelector(mlContext, validationDataView, trainDataView, availableColumns, dataProcessPipeline, Utils.FeatureSelection.BackwardsSelection, writeLogLine);

            dataProcessPipeline = dataProcessPipeline
                .Append(mlContext.Transforms.Concatenate("Features", columnsToUse.ToArray()))
                .AppendCacheCheckpoint(mlContext);

            // Set the training algorithm -- note: ideally, the final trainer would be fit on a fully independent dataset split from the submodels; I don't think this is simple when using the estimators API.

            // MicroAcc, MacroAcc, LogLossReduction
            // -----------------------------------
            // 0.9174, 0.5000, -0.609
            // var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features", numberOfIterations: 10), labelColumnName: "Label"); // todo: file bug that AveragedPerceptron does not expose a weight column

            // 0.8747, 0.5023, -7.692
            // var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features", exampleWeightColumnName: "Weight", minimumExampleCountPerLeaf: 2), labelColumnName: "Label");

            // 0.9034, 0.4975, -0.563
            //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label", featureColumnName: "Features", exampleWeightColumnName: "Weight"), labelColumnName: "Label");

            // 0.8431, 0.5975, -0.991
            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LdSvm(labelColumnName: "Label", featureColumnName: "Features", exampleWeightColumnName: "Weight"), labelColumnName: "Label");

            // 0.8700, 0.5151, -1.773
            //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Label", featureColumnName: "Features", exampleWeightColumnName: "Weight"), labelColumnName: "Label");

            // 0.6815, 0.6526, -3.095
            //var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(enforceNonNegativity: true, exampleWeightColumnName: "Weight", featureColumnName: "Features", labelColumnName: "Label");

            // 0.8607, 0.5151, -2.322
            //var trainer = mlContext.MulticlassClassification.Trainers.LightGbm(exampleWeightColumnName: "Weight", featureColumnName: "Features", labelColumnName: "Label");

            // 0.7948, 0.5814, -26.300
            //var trainer = mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(exampleWeightColumnName: "Weight", featureColumnName: "Features", labelColumnName: "Label");

            // 0.7846, 0.6526, -20.067
            //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"), labelColumnName: "Label"); // todo: file bug that SymbolicSgdLogisticRegression does not expose a weight column

            // 0.3881, 0.6409, -72.571
            //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SgdCalibrated(labelColumnName: "Label", featureColumnName: "Features", exampleWeightColumnName: "Weight"), labelColumnName: "Label");

            // 0.6379, 0.7004, -28.035 (no weight)
            //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SgdCalibrated(labelColumnName: "Label", featureColumnName: "Features"), labelColumnName: "Label");



            var trainingPipeline = dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

            return trainingPipeline;
        }


        public static ITransformer TrainModel(MLContext mlContext, IDataView trainDataView, IEstimator<ITransformer> trainingPipeline, Action<string> writeLogLine)
        {
            var sw = new Stopwatch();
            sw.Start();

            writeLogLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainDataView);

            writeLogLine($"=============== End of training process ({sw.ElapsedMilliseconds / 1000.0} sec) ===============\n");
            return model;
        }


        private static void EvaluateModel(MLContext mlContext, ITransformer mlModel, IDataView testDataView, Action<string> writeLogLine)
        {
            var sw = new Stopwatch();
            sw.Start();

            // Evaluate the model and show accuracy stats
            writeLogLine("===== Evaluating Model's accuracy with Test data =====");
            IDataView predictions = mlModel.Transform(testDataView);

            // Obtuse method of getting the number of classes
            VBuffer<ReadOnlyMemory<char>> classNamesVBuffer = default;
            predictions.Schema["Score"].GetSlotNames(ref classNamesVBuffer);
            var numClasses = classNamesVBuffer.Length;
            string[] classNames = classNamesVBuffer.DenseValues().Select(a => a.ToString()).ToArray();

            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score", topKPredictionCount: numClasses); // Todo: fix bug to allow for `topKPredictionCount: Int32.MaxValue` 
            ConsoleHelper.PrintMulticlassClassificationMetrics(metrics, classNames, writeLogLine);
            writeLogLine($"===== Finished Evaluating Model's accuracy with Test data ({sw.ElapsedMilliseconds / 1000.0} sec) =====");
        }


        private static ExperimentResult<MulticlassClassificationMetrics> TrainAutoMLSubPipeline1(MLContext mlContext, IDataView trainDataView, IDataView validationDataView, Action<string> writeLogLine)
        {
            var sw = new Stopwatch();
            sw.Start();

            ExperimentResult<MulticlassClassificationMetrics> experimentResult;

            //writeLogLine("\n=============== Training Stacked AutoML model on Text, Categorical, Numeric features ===============");
            writeLogLine("\n=============== Training Stacked AutoML model on Text features ==============="); // todo: print the metric being optimized

            Func<MulticlassClassificationMetrics> GetBaselineMetrics = () =>
            {
                var baselinePipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                    .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.Prior(labelColumnName: "Label", exampleWeightColumnName: "Weight"), labelColumnName: "Label"));
                var model = baselinePipeline.Fit(trainDataView);
                var predictions = model.Transform(validationDataView);
                var baselineMetrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");
                return baselineMetrics;
            };

            var progressHandler = new MulticlassExperimentProgressHandler(GetBaselineMetrics, writeLogLine);

            var experimentSettings = new MulticlassExperimentSettings
            {
                MaxExperimentTimeInSeconds = 7 * 60, //3600,
                //OptimizingMetric = MulticlassClassificationMetric.LogLossReduction,
                OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
                //OptimizingMetric = MulticlassClassificationMetric.MacroAccuracy,
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

            MLContext mlContextTmp = new MLContext(); // todo: log bug that AutoML is canceling the main MLContext as time expires
            mlContextTmp.Log += Program.ConsoleLogger;
            mlContextTmp.Log += Program.FileLogger;

            experimentResult = mlContextTmp.Auto()
                .CreateMulticlassClassificationExperiment(experimentSettings)
                .Execute(
                    trainData: trainDataView,
                    validationData: validationDataView,
                    progressHandler: progressHandler,
                    columnInformation: columnInformation);

            writeLogLine("\nBest run:");
            //progressHandler.Report(experimentResult.BestRun);
            var iteration = experimentResult.RunDetails.ToList().IndexOf(experimentResult.BestRun) + 1;
            ConsoleHelper.PrintIterationMetrics(iteration, experimentResult.BestRun.TrainerName, experimentResult.BestRun.ValidationMetrics, experimentResult.BestRun.RuntimeInSeconds, writeLogLine);
            writeLogLine($"=============== Finished training AutoML model ({sw.ElapsedMilliseconds / 1000.0} sec) ===============");

            return experimentResult;
        }


        private static ExperimentResult<MulticlassClassificationMetrics> TrainAutoMLSubPipeline2(MLContext mlContext, IDataView trainDataView, IDataView validationDataView, Action<string> writeLogLine)
        {
            var sw = new Stopwatch();
            sw.Start();

            ExperimentResult<MulticlassClassificationMetrics> experimentResult;

            writeLogLine("\n=============== Training Stacked AutoML model on Image features ===============");

            Func<MulticlassClassificationMetrics> GetBaselineMetrics = () =>
            {
                var baselinePipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                    .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.Prior(labelColumnName: "Label", exampleWeightColumnName: "Weight"), labelColumnName: "Label"));
                var model = baselinePipeline.Fit(trainDataView);
                var predictions = model.Transform(validationDataView);
                var baselineMetrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");
                return baselineMetrics;
            };

            var progressHandler = new MulticlassExperimentProgressHandler(GetBaselineMetrics, writeLogLine);

            var experimentSettings = new MulticlassExperimentSettings
            {
                MaxExperimentTimeInSeconds = 7 * 60, //40 * 60,
                //OptimizingMetric = MulticlassClassificationMetric.LogLossReduction,
                OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
                //OptimizingMetric = MulticlassClassificationMetric.MacroAccuracy,
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

            MLContext mlContextTmp = new MLContext(); // todo: log bug that AutoML is canceling the main MLContext as time expires
            mlContextTmp.Log += Program.ConsoleLogger;
            mlContextTmp.Log += Program.FileLogger;

            experimentResult = mlContextTmp.Auto()
                .CreateMulticlassClassificationExperiment(experimentSettings)
                .Execute(
                    //trainData: mlContext.Data.TakeRows(mlContext.Data.ShuffleRows(input: trainDataView, shufflePoolSize: 5000), 100),
                    trainData: trainDataView,
                    validationData: validationDataView,
                    progressHandler: progressHandler,
                    columnInformation: columnInformation //,
                                                         //preFeaturizer: preFeaturizer
                  );

            writeLogLine("\nBest run:");
            //progressHandler.Report(experimentResult.BestRun);
            var iteration = experimentResult.RunDetails.ToList().IndexOf(experimentResult.BestRun) + 1;
            ConsoleHelper.PrintIterationMetrics(iteration, experimentResult.BestRun.TrainerName, experimentResult.BestRun.ValidationMetrics, experimentResult.BestRun.RuntimeInSeconds, writeLogLine);
            writeLogLine($"=============== Finished training AutoML model ({sw.ElapsedMilliseconds / 1000.0} sec) ===============");

            return experimentResult;
        }


        private static ExperimentResult<MulticlassClassificationMetrics> TrainAutoMLSubPipeline3(MLContext mlContext, IDataView trainDataView, IDataView validationDataView, Action<string> writeLogLine)
        {
            var sw = new Stopwatch();
            sw.Start();

            ExperimentResult<MulticlassClassificationMetrics> experimentResult;

            writeLogLine("\n=============== Training Stacked AutoML model on Image features using TensorFlow ===============");

            // Currently unused
            Func<MulticlassClassificationMetrics> GetBaselineMetrics = () =>
            {
                var baselinePipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                    .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.Prior(labelColumnName: "Label", exampleWeightColumnName: "Weight"), labelColumnName: "Label"));
                var model = baselinePipeline.Fit(trainDataView);
                var predictions = model.Transform(validationDataView);
                var baselineMetrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");
                return baselineMetrics;
            };

            var progressHandler = new MulticlassExperimentProgressHandler(GetBaselineMetrics, writeLogLine);

            var experimentSettings = new MulticlassExperimentSettings
            {
                MaxExperimentTimeInSeconds = 0, // Finish after the 1st model
                //OptimizingMetric = MulticlassClassificationMetric.LogLossReduction,
                OptimizingMetric = MulticlassClassificationMetric.MicroAccuracy,
                //OptimizingMetric = MulticlassClassificationMetric.MacroAccuracy,
            };

            MLContext mlContextTmp = new MLContext(); // todo: log bug that AutoML is canceling the main MLContext as time expires
            mlContextTmp.Log += Program.ConsoleLogger;
            mlContextTmp.Log += Program.FileLogger;


            ColumnInferenceResults columnInference = mlContextTmp.Auto().InferColumns(trainFileName, groupColumns: false);
            ColumnInformation columnInformation = columnInference.ColumnInformation;

            writeLogLine($"\nBefore ignoring columns:");
            PrintColumnInformation(columnInformation, writeLogLine);

            // Set all columns to be ignored (if we don't explicitly list the column purpose for each column, it will be re-inferred and used)
            columnInformation.CategoricalColumnNames.ToList().ForEach(a => columnInformation.IgnoredColumnNames.Add(a));
            columnInformation.CategoricalColumnNames.Clear();
            columnInformation.NumericColumnNames.ToList().ForEach(a => columnInformation.IgnoredColumnNames.Add(a));
            columnInformation.NumericColumnNames.Clear();
            columnInformation.TextColumnNames.ToList().ForEach(a => columnInformation.IgnoredColumnNames.Add(a));
            columnInformation.TextColumnNames.Clear();
            columnInformation.ImagePathColumnNames.ToList().ForEach(a => columnInformation.IgnoredColumnNames.Add(a));
            columnInformation.ImagePathColumnNames.Clear();

            // Move Label, Weight, img1FileName out of ignored and to their correct purposes
            columnInformation.IgnoredColumnNames.Remove("Label");
            columnInformation.IgnoredColumnNames.Remove("Weight");
            columnInformation.IgnoredColumnNames.Remove("img1FileName");
            columnInformation.LabelColumnName = "Label";
            columnInformation.ExampleWeightColumnName = "Weight";
            columnInformation.ImagePathColumnNames.Add("img1FileName");

            writeLogLine($"\nAfter ignoring columns:");
            PrintColumnInformation(columnInformation, writeLogLine);

            var basepathEscaped = basepath + Path.DirectorySeparatorChar;

            if (Path.DirectorySeparatorChar == '\\')
                basepathEscaped = basepathEscaped.Replace("\\", "\\\\"); // todo: verify the escaping works on Windows

            
            
            // Add the basepath to the image filesnames
            string expression = $"x : concat(\"{basepathEscaped}\", x)";
            writeLogLine(expression);
            var preFeaturizer = mlContextTmp.Transforms.Expression("img1FileName", expression, new[] { "img1FileName" });

            experimentResult = mlContextTmp.Auto()
                .CreateMulticlassClassificationExperiment(experimentSettings)
                .Execute(
                    trainData: trainDataView,
                    validationData: validationDataView,
                    progressHandler: progressHandler,
                    columnInformation: columnInformation,
                    preFeaturizer: preFeaturizer
                  );

            writeLogLine("\nBest run:");
            //progressHandler.Report(experimentResult.BestRun);
            var iteration = experimentResult.RunDetails.ToList().IndexOf(experimentResult.BestRun) + 1;
            ConsoleHelper.PrintIterationMetrics(iteration, experimentResult.BestRun.TrainerName, experimentResult.BestRun.ValidationMetrics, experimentResult.BestRun.RuntimeInSeconds, writeLogLine);
            writeLogLine($"=============== Finished training AutoML model with TensorFlow ({sw.ElapsedMilliseconds / 1000.0} sec) ===============");

            return experimentResult;
        }

        private static void PrintColumnInformation(ColumnInformation columnInformation, Action<string> writeLogLine)
        {
            // Single valued
            writeLogLine($"Label: {columnInformation.LabelColumnName}");
            writeLogLine($"Weight: {columnInformation.ExampleWeightColumnName}");
            writeLogLine($"SamplingKey: {columnInformation.SamplingKeyColumnName}");
            writeLogLine($"GroupId: {columnInformation.GroupIdColumnName}");
            writeLogLine($"ItemId: {columnInformation.ItemIdColumnName}");
            writeLogLine($"UserId: {columnInformation.UserIdColumnName}");

            // Multi-valued
            writeLogLine($"Categorical: [{String.Join(", ", columnInformation.CategoricalColumnNames)}]");
            writeLogLine($"Numeric: [{String.Join(", ", columnInformation.NumericColumnNames)}]");
            writeLogLine($"Text: [{String.Join(", ", columnInformation.TextColumnNames)}]");
            writeLogLine($"Image: [{String.Join(", ", columnInformation.ImagePathColumnNames)}]");
            writeLogLine($"Ignored: [{String.Join(", ", columnInformation.IgnoredColumnNames)}]");
        }



        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema, Action<string> writeLogLine)
        {
            // Save/persist the trained model to a .ZIP file
            writeLogLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            writeLogLine($"The model is saved to {GetAbsolutePath(modelRelativePath)}"); // todo: print the model file size, perhaps also the unzipped size
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
        private readonly Action<string> WriteLogLine;

        public MulticlassExperimentProgressHandler(Func<MulticlassClassificationMetrics> getBaselineMetrics, Action<string> writeLogLine)
        {
            GetBaselineMetrics = getBaselineMetrics;
            WriteLogLine = writeLogLine;
        }

        public void Report(RunDetail<MulticlassClassificationMetrics> iterationResult)
        {
            if (_iterationIndex++ == 0)
            {
                ConsoleHelper.PrintMulticlassClassificationMetricsHeader(WriteLogLine);
           
                // todo: Disabled due to bug in OVA-PriorPredictor. See bug: https://github.com/dotnet/machinelearning/issues/5575
                // Print baseline metrics
                //var sw = new Stopwatch();
                //sw.Start();
                //var baselineMetrics = GetBaselineMetrics();
                //var runTime = sw.ElapsedMilliseconds / 1000.0;
                //ConsoleHelper.PrintIterationMetrics(-1, "Baseline", baselineMetrics, runTime);
            }

            if (iterationResult.Exception != null && !(iterationResult.Exception is OperationCanceledException))
                ConsoleHelper.PrintIterationException(iterationResult.Exception, WriteLogLine);
            else
                ConsoleHelper.PrintIterationMetrics(_iterationIndex, iterationResult.TrainerName,
                    iterationResult.ValidationMetrics, iterationResult.RuntimeInSeconds, WriteLogLine);
        }
    }

    internal static class ConsoleHelper
    {
        private const int Width = 114;

        public static void PrintMulticlassClassificationMetrics(MulticlassClassificationMetrics metrics, string[] classNames, Action<string> writer)
        {
            writer($"************************************************************");
            writer($"*    Metrics for multi-class classification model   ");
            writer($"*-----------------------------------------------------------");
            writer($"Accuracy (micro-avg):              {metrics.MicroAccuracy:0.0000}   # 0..1, higher is better");
            writer($"Accuracy (macro):                  {metrics.MacroAccuracy:0.0000}   # 0..1, higher is better");
            writer($"Top-K accuracy:                    [{string.Join(", ", metrics?.TopKAccuracyForAllK?.Select(a => $"{a:0.0000}") ?? new string[] { "Set topKPredictionCount in evaluator to view" })}]   # 0..1, higher is better");
            writer($"Log-loss reduction:                {metrics.LogLossReduction:0.0000;-0.000}   # -Inf..1, higher is better");
            writer($"Log-loss:                          {metrics.LogLoss:0.0000}   # 0..Inf, lower is better");
            writer("\nPer class metrics");
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                writer($"LogLoss for class {i} ({classNames[i] + "):",-11}   {metrics.PerClassLogLoss[i]:0.0000}   # 0..Inf, lower is better");
            }
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                writer($"Precision for class {i} ({classNames[i] + "):",-11} {metrics.ConfusionMatrix.PerClassPrecision[i]:0.0000}   # 0..1, higher is better");
            }
            for (int i = 0; i < metrics.PerClassLogLoss.Count; i++)
            {
                writer($"Recall for class {i} ({classNames[i] + "):",-11}    {metrics.ConfusionMatrix.PerClassRecall[i]:0.0000}   # 0..1, higher is better");
            }
            writer("");
            writer(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            writer($"************************************************************");
        }


        internal static void PrintIterationMetrics(int iteration, string trainerName, MulticlassClassificationMetrics metrics, double? runtimeInSeconds, Action<string> writer)
        {
            CreateRow($"{iteration,-4} {trainerName,-35} {metrics?.MicroAccuracy ?? double.NaN,14:F4} {metrics?.MacroAccuracy ?? double.NaN,14:F4} {metrics?.LogLossReduction ?? double.NaN,17:F4} {runtimeInSeconds.Value,9:F1}", Width, writer);
        }

        internal static void PrintIterationException(Exception ex, Action<string> writer)
        {
            writer($"Exception during AutoML iteration: {ex}");
        }

        internal static void PrintMulticlassClassificationMetricsHeader(Action<string> writer)
        {
            CreateRow($"{"",-4} {"Trainer",-35} {"MicroAccuracy",14} {"MacroAccuracy",14} {"LogLossReduction",17} {"Duration",9}", Width, writer);
        }

        private static void CreateRow(string message, int width, Action<string> writer)
        {
            var threadHeader = (string.IsNullOrEmpty(Thread.CurrentThread.Name) ? "" : Thread.CurrentThread.Name + ": ");
            writer(threadHeader + "|" + message.PadRight(width - 2) + "|");
        }

        public static void PrintMulticlassClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValResults, Action<string> writer)
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

            writer($"*************************************************************************************************************");
            writer($"*       Metrics for Multi-class Classification model      ");
            writer($"*------------------------------------------------------------------------------------------------------------");
            writer($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            writer($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            writer($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            writer($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            writer($"*************************************************************************************************************");
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
