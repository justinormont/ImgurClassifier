//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using Microsoft.ML.Data;

namespace ImgurClassifier.Model.DataModels
{
    public class ModelInput
    {
        [ColumnName("Label"), LoadColumn(0)]
        public string Label { get; set; }


        [ColumnName("Weight"), LoadColumn(1)]
        public float Weight { get; set; }


        [ColumnName("Name"), LoadColumn(2)]
        public string Name { get; set; }


        [ColumnName("title"), LoadColumn(3)]
        public string Title { get; set; }


        [ColumnName("tags"), LoadColumn(4)]
        public string Tags { get; set; }


        [ColumnName("tagCount"), LoadColumn(5)]
        public float TagCount { get; set; }


        [ColumnName("tagAvgFollowers"), LoadColumn(6)]
        public float TagAvgFollowers { get; set; }


        [ColumnName("tagSumFollowers"), LoadColumn(7)]
        public float TagSumFollowers { get; set; }


        [ColumnName("tagAvgTotalItems"), LoadColumn(8)]
        public float TagAvgTotalItems { get; set; }


        [ColumnName("tagSumTotalItems"), LoadColumn(9)]
        public float TagSumTotalItems { get; set; }


        [ColumnName("imagesCount"), LoadColumn(10)]
        public float ImagesCount { get; set; }


        [ColumnName("img1Uri"), LoadColumn(11)]
        public string Img1Uri { get; set; }


        [ColumnName("img1Desc"), LoadColumn(12)]
        public string Img1Desc { get; set; }


        [ColumnName("img1Type"), LoadColumn(13)]
        public string Img1Type { get; set; }


        [ColumnName("img1FileName"), LoadColumn(14)]
        public string Img1FileName { get; set; }


        [ColumnName("img2Uri"), LoadColumn(15)]
        public string Img2Uri { get; set; }


        [ColumnName("img2Desc"), LoadColumn(16)]
        public string Img2Desc { get; set; }


        [ColumnName("img2Type"), LoadColumn(17)]
        public string Img2Type { get; set; }


        [ColumnName("img2FileName"), LoadColumn(18)]
        public string Img2FileName { get; set; }


        [ColumnName("img3Uri"), LoadColumn(19)]
        public string Img3Uri { get; set; }


        [ColumnName("img3Desc"), LoadColumn(20)]
        public string Img3Desc { get; set; }


        [ColumnName("img3Type"), LoadColumn(21)]
        public string Img3Type { get; set; }


        [ColumnName("img3FileName"), LoadColumn(22)]
        public string Img3FileName { get; set; }


    }
}
