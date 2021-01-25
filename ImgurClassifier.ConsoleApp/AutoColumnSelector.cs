using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;

namespace ImgurClassifier.Extras
{
    public static partial class Utils
    {
        // Feature selection on the column level to choose the best combination of columns
        public static IEnumerable<string> AutoColumnSelector(MLContext mlContext, IDataView validationDataView, IDataView trainingDataView, IEnumerable<string> columnNames, IEstimator<ITransformer> trainingPipeline, FeatureSelection searchPattern, Action<string> writeLogLine)
        {
            var sw = new Stopwatch();
            sw.Start();

            writeLogLine("=============== Sweeping featurization columns ===============");

            var columnList = columnNames.ToArray();
            var columnsToPrefetch = columnNames.Union(new[] { "Label", "Weight" }).ToArray();

            var fitPipeline = trainingPipeline.Fit(trainingDataView);
            var transformedTrainingDataView = mlContext.Data.Cache(fitPipeline.Transform(trainingDataView), columnsToPrefetch: columnsToPrefetch);
            var transformedValidationDataView = mlContext.Data.Cache(fitPipeline.Transform(validationDataView), columnsToPrefetch: columnsToPrefetch);

            //var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features", numberOfIterations: 10), labelColumnName: "Label");
            var trainer = mlContext.MulticlassClassification.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features");
            HashSet<string> bestSetOfColumns = new();
            double bestMetric = double.NaN;

            writeLogLine($"{"",11} {"MicroAccuracy",14} {"MacroAccuracy",14} {"LogLossReduction",17} {"Duration",9} TrainingColumns");

            // Define action to use within the switch statement below
            Func<string[], int, string, (double, double)> actionTrainOnColumnsAndUpdateBest = (string[] columnsForIteration, int i, string total) =>
            {
                var sw2 = new Stopwatch();
                sw2.Start();
                double gain = double.NaN;
                double metric = double.NaN;
                try
                {
                    var pipelineForIteration = mlContext.Transforms.Concatenate("Features", columnsForIteration).Append(trainer);
                    var model = pipelineForIteration.Fit(transformedTrainingDataView);
                    var predictions = model.Transform(transformedValidationDataView);
                    var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");
                //metric = Math.Sqrt(metrics.MicroAccuracy * metrics.MacroAccuracy); // Geometric mean
                metric = metrics.MicroAccuracy * 0.8 + metrics.MacroAccuracy * 0.2; // Weighted arithmetic mean
                gain = metric - (!double.IsNaN(bestMetric) ? bestMetric : double.NegativeInfinity); // Assumes higher is better

                if (metric > bestMetric || double.IsNaN(bestMetric))
                    {
                        writeLogLine($"New leader: {metric:F4}");
                        bestMetric = metric;
                        bestSetOfColumns = new HashSet<string>(columnsForIteration);
                    }
                // todo: print in binary form to help users understand the search pattern -- Convert.ToString(i, 2).PadLeft(columnCount, '0');
                writeLogLine($"{i + " of " + total,11} {metrics?.MicroAccuracy ?? double.NaN,14:F4} {metrics?.MacroAccuracy ?? double.NaN,14:F4} {metrics?.LogLossReduction ?? double.NaN,17:F4} {sw2.ElapsedMilliseconds / 1000.0,9:F1} cols=[{string.Join(", ", columnsForIteration)}]");
                }
                catch (Exception e)
                {
                    writeLogLine($"Iteration {i} failed. cols=[{string.Join(", ", columnsForIteration)}] duration={sw2.ElapsedMilliseconds / 1000.0,9:F1} error={e}");
                }
                return (gain, metric);
            };


            switch (searchPattern)
            {
                case FeatureSelection.ExhaustiveSearch: // O(2^N); boolean combinatorics of each column on/off
                    {
                        var count = Math.Pow(2, columnList.Length);
                        for (var i = 1; i < count; i++) // i=0 would be no columns, so start at 1
                        {
                            var columnsForIteration = columnList.Where((_, j) => ((i >> j) & 1) == 1).ToArray(); // Bit twiddling; shift and mask to read bits
                            actionTrainOnColumnsAndUpdateBest(columnsForIteration, i, count.ToString());
                        }
                        break;
                    }

                case FeatureSelection.RandomSearch: // O(C); try random subsets of the columns w/ early stopping
                    {
                        var earlyStoppingRounds = 50;
                        var count = earlyStoppingRounds;
                        var rand = new Random();
                        var max = (int)Math.Pow(2, columnList.Length);
                        var tried = new HashSet<int>();
                        actionTrainOnColumnsAndUpdateBest(columnList, 0, count.ToString() + "+"); // Baseline with all columns

                        for (var k = 1; k <= count; k++) // Iteration 0 is used for the baseline run, so start at 1
                        {
                            int i, randAttempts = 0;

                            do
                            {
                                i = rand.Next(1, max); // i=0 would be no columns, so start at 1
                                randAttempts++;
                            }
                            while (tried.Contains(i) && randAttempts < 10);

                            if (randAttempts == 10)
                                break; // Cound not find an new set of columns to try

                            var columnsForIteration = columnList.Where((_, j) => ((i >> j) & 1) == 1).ToArray(); // Bit twiddling; shift and mask to read bits
                            var (gain, _) = actionTrainOnColumnsAndUpdateBest(columnsForIteration, k, count.ToString() + "+");
                            if (gain > 0)
                                count = k + 50; // Extend limit for early stopping
                        }
                        break;
                    }

                case FeatureSelection.OnePassRemoval: // O(N); tries removing each column; remove if no improvement to metrics
                    {
                        var removedColumns = new HashSet<string>();
                        actionTrainOnColumnsAndUpdateBest(columnList, 0, columnList.Length.ToString()); // Baseline to set bestMetric

                        for (var i = 0; i < columnList.Length; i++)
                        {
                            var columnsForIteration = columnList.Where((col, j) => (j != i && !removedColumns.Contains(col))).ToArray();
                            var (gain, metric) = actionTrainOnColumnsAndUpdateBest(columnsForIteration, i + 1, (columnList.Length + 1).ToString());
                            if (gain > 0) // If better without the column, remove it
                                removedColumns.Add(columnList[i]);
                        }
                        break;
                    }

                case FeatureSelection.ForwardSelection: // O(N^2); iteratively add the next best column after each pass
                    {
                        double bestMetricOverall = double.NegativeInfinity;
                        HashSet<string> currentSetOfColumns = new();
                        string bestNextColumn;
                        int count = 0;
                        actionTrainOnColumnsAndUpdateBest(columnList, 0, columnList.Length.ToString()); // Baseline with all columns

                        do
                        {
                            bestNextColumn = null;
                            int countBeforePass = count;
                            for (var i = 0; i < columnList.Length; i++)
                            {
                                if (currentSetOfColumns.Contains(columnList[i]))
                                    continue; // No need to try this column, it's already selected
                                var columnsForIteration = columnList.Where((col, j) => (j == i || currentSetOfColumns.Contains(col))).ToArray();
                                var (gain, metric) = actionTrainOnColumnsAndUpdateBest(columnsForIteration, count++, $"<={Math.Ceiling(Math.Pow(columnList.Length - currentSetOfColumns.Count(), 2) / 2) + countBeforePass}");
                                if (metric > bestMetricOverall)
                                {
                                    bestMetricOverall = metric;
                                    bestNextColumn = columnList[i];
                                }
                            }

                            if (bestNextColumn != null)
                                currentSetOfColumns.Add(bestNextColumn);
                        }
                        while (bestNextColumn != null);
                        break;
                    }

                case FeatureSelection.BackwardsSelection: // O(N^2); iteratively remove the next worst column after each pass
                    {
                        double bestMetricOverall = double.NegativeInfinity;
                        HashSet<string> currentSetOfColumns = new(columnList);
                        string worstNextColumn;
                        int count = 0;
                        actionTrainOnColumnsAndUpdateBest(columnList, count++, $"<={Math.Ceiling(Math.Pow(columnList.Length, 2) / 2)}"); // Baseline to set bestMetric

                        do
                        {
                            worstNextColumn = null;
                            int countBeforePass = count;
                            for (var i = 0; i < columnList.Length; i++)
                            {
                                if (!currentSetOfColumns.Contains(columnList[i]))
                                    continue; // No need to try this column, it's already selected
                                var columnsForIteration = columnList.Where((col, j) => (j != i && currentSetOfColumns.Contains(col))).ToArray();
                                var (gain, metric) = actionTrainOnColumnsAndUpdateBest(columnsForIteration, count++, $"<={Math.Ceiling(Math.Pow(columnList.Length, 2) / 2)}");
                                if (metric >= bestMetricOverall)
                                {
                                    bestMetricOverall = metric;
                                    worstNextColumn = columnList[i];
                                }
                            }

                            if (worstNextColumn != null)
                                currentSetOfColumns.Remove(worstNextColumn);
                        }
                        while (worstNextColumn != null);
                        break;
                    }
                default:
                    throw new Exception($"Unknown search scheme: {searchPattern}");
            }

            var selectedColumns = columnList.Where(col => bestSetOfColumns.Contains(col));
            var nonSelectedColumns = columnList.Where(col => !bestSetOfColumns.Contains(col));

            writeLogLine($"Removed columns: [{string.Join(", ", nonSelectedColumns)}]");
            writeLogLine($"Best set of columns: [{string.Join(", ", selectedColumns)}]");
            writeLogLine($"=============== End of sweeping featurization columns ({sw.ElapsedMilliseconds / 1000.0} sec) ===============\n");

            return selectedColumns;
        }

        public enum FeatureSelection
        {
            ExhaustiveSearch = 1,
            OnePassRemoval = 2,
            ForwardSelection = 3,
            BackwardsSelection = 4,
            RandomSearch = 5,
        }
    }
}