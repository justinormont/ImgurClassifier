using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;

namespace ImgurClassifier.Extras
{
    public partial class Utils
    {
        public enum Task
        {
            MulticlassClassification = 1,
            BinaryClassification = 2,
            Regression = 3,
        }

        // Trains a proxy model (FastForest)
        public static void ProxyModelFeatureImportance(MLContext mlContext, Task task, string labelColumnName, string featureColumnName, string exampleWeightColumnName, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline, Action<string> writeLogLine)
        {
            var trainerOptions = new FastForestRegressionTrainer.Options
            {
                //FeatureFirstUsePenalty = 0.1,
                NumberOfLeaves = 20,
                FeatureFraction = 0.7,
                NumberOfTrees = 200,
                LabelColumnName = "FloatLabel",
                FeatureColumnName = featureColumnName,
                ExampleWeightColumnName = exampleWeightColumnName,
                //ExecutionTime = true,

                // Shuffle the label ordering before each tree is learned.
                // Needed when running a multi-class dataset as regression.
                // We are only interested in the split gains in the trees,
                // and not outputting correct classes, so a regression tree
                // works for all tasks.
                ShuffleLabels = task == Task.MulticlassClassification,
            };

            // Define the tree-based featurizer's configuration.
            /*var options = new FastForestRegressionFeaturizationEstimator.Options
            {
                InputColumnName = featureColumnName,
                TreesColumnName = "FeaturesTreeFeatTrees",
                LeavesColumnName = "FeaturesTreeFeatLeaves",
                PathsColumnName = "FeaturesTreeFeatPaths",
                TrainerOptions = trainerOptions
            };*/

            Action<RowWithKey, RowWithFloat> actionConvertKeyToFloat = (RowWithKey rowWithKey, RowWithFloat rowWithFloat) =>
            {
                rowWithFloat.FloatLabel = rowWithKey.Label == 0 ? float.NaN : rowWithKey.Label - 1;
            };

            var pipeline = trainingPipeline;

            switch (task)
            {
                case Task.MulticlassClassification:
                    if (labelColumnName != "Label")
                    {
                        // The below actionConvertKeyToFloat expects the input column name to be "Label"
                        pipeline = pipeline.Append(mlContext.Transforms.CopyColumns("Label", labelColumnName));
                    }

                    // Convert the Key type label to a Float (so we can use a regression trainer)
                    //pipeline = pipeline.Append(mlContext.Transforms.CustomMapping(actionConvertKeyToFloat, contractName: null));
                    pipeline = pipeline.Append(mlContext.Transforms.CustomMapping(new ConvertLabelKeyToFloat().GetMapping(), "ConvertLabelKeyToFloat"));
                    break;

                case Task.BinaryClassification:
                    // Convert the Boolean type label to a Float (so we can use a regression trainer)
                    pipeline = pipeline.Append(mlContext.Transforms.Expression("FloatLabel", "x : (x ? 1.0 : 0.0)", new[] { labelColumnName }));
                    break;

                case Task.Regression:
                    // Convert the Boolean type label to a Float (so we can use a regression trainer)
                    pipeline = pipeline.Append(mlContext.Transforms.CopyColumns("FloatLabel", labelColumnName));
                    break;

                default:
                    throw new NotImplementedException($"Unknown task: {task}");
            }

            // Train a FastForestRegression model
            //var finalPipeline = pipeline.Append(mlContext.Transforms.FeaturizeByFastForestRegression(options));
            var finalPipeline = pipeline.Append(mlContext.Regression.Trainers.FastForest(trainerOptions));

            var sw = new Stopwatch();
            sw.Start();
            writeLogLine("=============== Training proxy model ===============");

            // Fit this pipeline to the training data.
            var model = finalPipeline.Fit(trainingDataView);

            writeLogLine($"=============== End of proxy training process ({sw.ElapsedMilliseconds / 1000.0} sec) ===============\n");

            // Get the feature importance based on the information gain used during training.
            VBuffer<float> weights = default;
            model.LastTransformer.Model.GetFeatureWeights(ref weights);
            float[] weightsValues = weights.DenseValues().ToArray();

            // Get the name of the features (slot names)
            var output = model.Transform(trainingDataView);
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            output.Schema[featureColumnName].GetSlotNames(ref slotNames);

            // Sort to place the most important features first
            IEnumerable<string> slotWeightText = slotNames.Items()
                .Select((kvp, slotIndex) =>
                    (
                        featureName: $"{(kvp.Value.Length > 0 ? kvp.Value : $"UnnamedSlot_{slotIndex:000000}")}",
                        featureImportance: weightsValues[slotIndex],
                        featureImportanceAbs: (float)Math.Abs(weightsValues[slotIndex])
                    )
                )
                .Where(tuple => tuple.featureImportanceAbs > 0)
                .OrderByDescending(tuple => tuple.featureImportanceAbs)
                .Take(100)
                .Select((tuple, featureImportanceIndex) => $"{featureImportanceIndex,-3} {tuple.featureImportance,-14}: {tuple.featureName}");

            writeLogLine($"\nFeature importance: (top {Math.Min(100, weightsValues.Length):n0} of {weightsValues.Length:n0})");
            writeLogLine(String.Join("\n", slotWeightText));
        }


        #region ConvertLabelKeyToFloat CustomMapping
        [CustomMappingFactoryAttribute("ConvertLabelKeyToFloat")]
        public class ConvertLabelKeyToFloat : CustomMappingFactory<RowWithKey, RowWithFloat>
        {
            private static Action<RowWithKey, RowWithFloat> CustomAction = (RowWithKey rowWithKey, RowWithFloat rowWithFloat) =>
            {
                rowWithFloat.FloatLabel = rowWithKey.Label == 0 ? float.NaN : rowWithKey.Label - 1;
            };

            public override Action<RowWithKey, RowWithFloat> GetMapping() => CustomAction;
        }

        public class RowWithKey
        {
            [KeyType(100000)] // Allows up to 100k classes
            public uint Label { get; set; }
        }

        public class RowWithFloat
        {
            public float FloatLabel { get; set; }
        }
        #endregion
    }
}
