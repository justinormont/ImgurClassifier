using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace ImgurClassifier.ConsoleApp
{
    public class StringStatisticsFeaturizer
    {
        #region StringStatistics CustomMapping
        [CustomMappingFactoryAttribute("StringStatistics")]
        public class StringStatisticsAction : CustomMappingFactory<RowWithText, RowWithStringStatistics>
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

        public class RowWithText
        {
            [ColumnName("text")] // Harded coded to use an input column named "text"
            public string Text { get; set; }
        }

        public class RowWithStringStatistics
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
    }
}
