using Microsoft.ML.Data;

namespace ImgurClassifier.Model.DataModels
{
    public class ModelInput
    {
        [ColumnName("Label"), LoadColumn(0)]
        public string Label { get; set; }

        // Instance weight based on the label value; to up-weight the user submission
        // Note: Leaks the label
        [ColumnName("Weight"), LoadColumn(1)]
        public float Weight { get; set; }


        // Post ID (used as a name for the row to assist in local explainability)
        [ColumnName("Name"), LoadColumn(2)]
        public string Name { get; set; }


        // Post title
        [ColumnName("title"), LoadColumn(3)]
        public string Title { get; set; }


        // Tags on the post. E.g. "cats - dogs - aww"
        [ColumnName("tags"), LoadColumn(4)]
        public string Tags { get; set; }


        // Number of tags used on the post
        [ColumnName("tagCount"), LoadColumn(5)]
        public float TagCount { get; set; }


        // Average number followers of the tags
        [ColumnName("tagAvgFollowers"), LoadColumn(6)]
        public float TagAvgFollowers { get; set; }


        // Sum of the followers of the tags
        [ColumnName("tagSumFollowers"), LoadColumn(7)]
        public float TagSumFollowers { get; set; }


        // Max followers of the tags
        [ColumnName("tagMaxFollowers"), LoadColumn(8)]
        public float TagMaxFollowers { get; set; }


        // Average number of posts using the tags
        [ColumnName("tagAvgTotalItems"), LoadColumn(9)]
        public float TagAvgTotalItems { get; set; }


        // Sum of posts using the tags
        [ColumnName("tagSumTotalItems"), LoadColumn(10)]
        public float TagSumTotalItems { get; set; }


        // Max posts using the tags
        [ColumnName("tagMaxTotalItems"), LoadColumn(11)]
        public float TagMaxTotalItems { get; set; }


        // Number of images in the post
        [ColumnName("imagesCount"), LoadColumn(12)]
        public float ImagesCount { get; set; }


        // URI for the 1st image
        [ColumnName("img1Uri"), LoadColumn(13)]
        public string Img1Uri { get; set; }


        // Description of the 1st image
        [ColumnName("img1Desc"), LoadColumn(14)]
        public string Img1Desc { get; set; }


        // MEME type of the 1st image -- e.g. video/mp4
        [ColumnName("img1Type"), LoadColumn(15)]
        public string Img1Type { get; set; }


        // Local relative path of downloaded image
        [ColumnName("img1FileName"), LoadColumn(16)]
        public string Img1FileName { get; set; }


        // URI for the 2nd image
        [ColumnName("img2Uri"), LoadColumn(17)]
        public string Img2Uri { get; set; }


        // Description of the 2nd image
        [ColumnName("img2Desc"), LoadColumn(18)]
        public string Img2Desc { get; set; }


        // MEME type of the 2nd image -- e.g. image/jpeg
        [ColumnName("img2Type"), LoadColumn(19)]
        public string Img2Type { get; set; }


        // Local relative path of downloaded image
        [ColumnName("img2FileName"), LoadColumn(20)]
        public string Img2FileName { get; set; }


        // URI for the 3rd image
        [ColumnName("img3Uri"), LoadColumn(21)]
        public string Img3Uri { get; set; }


        // Description of the 3rd image
        [ColumnName("img3Desc"), LoadColumn(22)]
        public string Img3Desc { get; set; }


        // MEME type of the 2nd image -- e.g. image/png
        [ColumnName("img3Type"), LoadColumn(23)]
        public string Img3Type { get; set; }


        // Local relative path of downloaded image
        [ColumnName("img3FileName"), LoadColumn(24)]
        public string Img3FileName { get; set; }


        // Unixtime of when the post was created -- e.g. 1609317658
        // Note: Can leak due to dataset collection date differences
        [ColumnName("datetime"), LoadColumn(25)]
        public string DateTime { get; set; }


        // Number of views the post has received (at dataset creation time)
        // Note: Leaks the label (can be used as an alt-label
        [ColumnName("views"), LoadColumn(26)]
        public float Views { get; set; }


        // Number of up-votes the post has received (at dataset creation time)
        // Note: Leaks the label (can be used as an alt-label
        [ColumnName("ups"), LoadColumn(27)]
        public float Ups { get; set; }


        // Number of down-votes the post has received (at dataset creation time)
        // Note: Leaks the label (can be used as an alt-label
        [ColumnName("downs"), LoadColumn(28)]
        public float Downs { get; set; }


        // Number of up-votes minus down-votes the post has received (at dataset creation time)
        // Note: Leaks the label (can be used as an alt-label
        [ColumnName("points"), LoadColumn(29)]
        public float Points { get; set; }


        // Imgur's assign score for the post
        // Note: Leaks the label (can be used as an alt-label
        [ColumnName("postScore"), LoadColumn(30)]
        public float PostScore { get; set; }


        // Number of comments on the post
        // Note: Leaks the label (can be used as an alt-label
        [ColumnName("commentCount"), LoadColumn(31)]
        public float CommentCount { get; set; }


        // Number of users that favorited the post
        // Note: Leaks the label (can be used as an alt-label
        [ColumnName("favoriteCount"), LoadColumn(32)]
        public float FavoriteCount { get; set; }
    }
}
