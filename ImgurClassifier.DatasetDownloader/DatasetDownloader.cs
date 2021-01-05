using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using Newtonsoft.Json.Linq;

namespace ImgurClassifier.DatasetDownloader
{
    public static class DatasetDownloader
    {

        private static List<(string split, float weight, string url)> urls = new(new (string split, float weight, string url)[]{
            // Recent user submissions
            /*("train", 1, "https://api.imgur.com/3/gallery/user/time/130?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/user/time/120?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/user/time/110?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/user/time/100?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/user/time/90?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/user/time/80?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/user/time/70?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/user/time/60?client_id="),
            ("valid", 1, "https://api.imgur.com/3/gallery/user/time/50?client_id="),
            ("valid", 1, "https://api.imgur.com/3/gallery/user/time/40?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/user/time/30?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/user/time/20?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/user/time/10?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/user/time/0?client_id="),*/

            // Random FrontPage images
            ("train", 1, "https://api.imgur.com/3/gallery/random/random/11?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/random/random/10?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/random/random/9?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/random/random/8?client_id="),
            ("valid", 1, "https://api.imgur.com/3/gallery/random/random/7?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/random/random/6?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/random/random/5?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/random/random/4?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/random/random/3?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/random/random/2?client_id="),
            ("valid", 1, "https://api.imgur.com/3/gallery/random/random/1?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/random/random/0?client_id="),

            // Recent hot images
            /*("train", 1, "https://api.imgur.com/3/gallery/hot/time/13?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/hot/time/12?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/hot/time/11?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/hot/time/10?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/hot/time/9?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/hot/time/8?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/hot/time/7?client_id="),
            ("train", 1, "https://api.imgur.com/3/gallery/hot/time/6?client_id="),
            ("valid", 1, "https://api.imgur.com/3/gallery/hot/time/5?client_id="),
            ("valid", 1, "https://api.imgur.com/3/gallery/hot/time/4?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/hot/time/3?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/hot/time/2?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/hot/time/1?client_id="),
            ("test", 1, "https://api.imgur.com/3/gallery/hot/time/0?client_id="),*/
        });

        public static Dictionary<string, string> GetDataset()
        {
            //List<(string split, float weight, string url)> urls = new();

            var r = new Random();
            int userSubCount = 300;
            for (var i = 30; i < userSubCount; i++)
            {
                string split;
                //if ((float)i / userSubCount < 0.3) // Newest posts in tests; oldest in train
                //    split = "test";
                //else if ((float)i / userSubCount < 0.5)
                //    split = "valid";
                //else
                //    split = "train";

                var s = r.NextDouble();
                if (s < 0.3)
                    split = "test";
                else if (s < 0.5)
                    split = "valid";
                else
                    split = "train";

                urls.Add((split, 1, $"https://api.imgur.com/3/gallery/user/time/{i}?client_id="));
            }

            /*int frontPageCount = 15;
            for (var i = 0; i < frontPageCount; i++)
            {
                string split;
                if ((float)i / frontPageCount < 0.3) // Newest posts in tests; oldest in train
                    split = "test";
                else if ((float)i / frontPageCount < 0.5)
                    split = "valid";
                else
                    split = "train";

                urls.Add((split, 1, $"https://api.imgur.com/3/gallery/hot/time/{i}?client_id="));
            }
            */

            return GetDataset(urls.ToArray());
        }

        private static Dictionary<string, string> GetDataset((string split, float weight, string url)[] urls)
        {
            var datasetSplits = new Dictionary<string, string>();
            var postsSeen = new HashSet<string>();

            var dataDirectoryName = "DataDir";
            Directory.CreateDirectory(dataDirectoryName);

            datasetSplits["basepath"] = Path.GetFullPath(".");

            // Create empty datasets w/ a header row
            foreach ((string split, float weight, string url) in urls)
            {
                var fileName = Path.Combine(dataDirectoryName, $"Imgur_{split}.tsv");
                datasetSplits[split] = fileName;
            }

            // If exists, return existing dataset
            if (datasetSplits.Select(a => a.Value).All(f => File.Exists(f) || Directory.Exists(f)))
            {
                return datasetSplits;
            }

            Mutex mutex = new Mutex(false, "ImgurClassifierDatasetDownloader");

            try
            {
                mutex.WaitOne();

                // If exists, return existing dataset
                if (datasetSplits.Select(a => a.Value).All(f => File.Exists(f) || Directory.Exists(f)))
                {
                    return datasetSplits;
                }

                Console.WriteLine("=============== Downloading dataset ===============");

                // Create empty datasets w/ a header row
                foreach ((string split, float weight, string url) in urls)
                {
                    using (var fs = File.CreateText(datasetSplits[split] + ".tmp"))
                    {
                        fs.WriteLine(string.Join('\t', new[] {
                            "Label",
                            "Weight",
                            "Name",
                            "title",
                            "tags",
                            "tagCount",
                            "tagAvgFollowers",
                            "tagSumFollowers",
                            "tagMaxFollowers",
                            "tagAvgTotalItems",
                            "tagSumTotalItems",
                            "tagMaxTotalItems",
                            "imagesCount",
                            "img1Uri",
                            "img1Desc",
                            "img1Type",
                            "img1FileName",
                            "img2Uri",
                            "img2Desc",
                            "img2Type",
                            "img2FileName",
                            "img3Uri",
                            "img3Desc",
                            "img3Type",
                            "img3FileName",
                            "datetime",
                            "views",
                            "ups",
                            "downs",
                            "points",
                            "postScore",
                            "commentCount",
                            "favoriteCount"
                        }));
                    }
                }

                // Generate a nonce like "3a22db833f45c0c"
                var r = new Random();
                string clientId = "546c25a59c58ad7"; // (r.Next().ToString("X") + r.Next().ToString("X")).ToLowerInvariant();

                foreach ((string split, float weight, string url) in urls)
                {
                    using (var fs = File.AppendText(datasetSplits[split] + ".tmp"))
                    {
                        Console.WriteLine($"Downloading {url + clientId}");
                        string json = new WebClient().DownloadString(url + clientId);
                        var jsonDict = JObject.Parse(json);
                        var rowsWritten = 0;
                        /*
                        f.Name = y.link;
                        f.title = y.title;
                        f.tags = y.tags.map(a => a.display_name).join(' - ');
                        f.tagCount = y.tags.length;

                        f.tagAvgFollowers = y.tags.map(a => a.followers).reduce((s, v, _, a) => v/a.length + s);

                        f.tagSumFollowers = y.tags.map(a => a.followers).reduce((s, v, _, a) => v + s);
                        f.tagAvgTotalItems = y.tags.map(a => a.total_items).reduce((s, v, _, a) => v/a.length + s);
                        f.tagSumTotalItems = y.tags.map(a => a.total_items).reduce((s, v, _, a) => v + s);
                        f.imagesCount = y.images_count;
                        f.img1Uri = y.images[0]?.link;
                        f.img1Desc = y.images[0]?.description;
                        f.img1Type = y.images[0]?.type;
                        */

                        var length = jsonDict["data"].Count();
                        Console.WriteLine($"split={split}: Found [{length}] posts for url {url}");

                        for (int i = 0; i < length; i++)
                        {
                            var row = jsonDict["data"][i];
                            bool isAlbum = (bool)(row?["is_album"]);

                            var postUrl = (string)row?["link"] ?? "";
                            if (postsSeen.Contains(postUrl))
                            {
                                //throw new Exception($"Post already seen: {postUrl}");
                                Console.WriteLine($"Post already seen: {postUrl}");
                                continue; // Ensure we don't have duplicate rows
                            }
                            postsSeen.Add(postUrl);

                            Console.WriteLine($"Time stamp: {DateTimeOffset.FromUnixTimeSeconds((long)row?["datetime"]).DateTime}");

                            List<string> features = new List<string>();

                            string label = ((bool)row["in_most_viral"] ? "FrontPage" : "UserSub");

                            string img1FileName, img2FileName, img3FileName;
                            try
                            {
                                img1FileName = DownloadImage(split, label, (string)(isAlbum ? row?["images"]?.ElementAtOrDefault(0)?["link"] : row?["link"]));
                                img2FileName = DownloadImage(split, label, (string)(row?["images"]?.ElementAtOrDefault(1)?["link"]));
                                img3FileName = DownloadImage(split, label, (string)(row?["images"]?.ElementAtOrDefault(2)?["link"]));

                                if (string.IsNullOrEmpty(img1FileName))
                                {
                                    continue; // If the main image failed to parse, don't write the dataset row
                                }
                            }
                            catch (WebException e)
                            {
                                Console.Error.WriteLine(e);
                                continue; // If an image failed to download, don't write the dataset row
                            }

                            // Label
                            features.Add(label);                             // Label
                            features.Add(weight.ToString());                 // Weight

                            // Post features
                            features.Add((string)row?["link"] ?? "");        // Name
                            features.Add(CleanText(row?["title"]) ?? "");    // title

                            // Tag features
                            features.Add(string.Join(" - ", row?["tags"].Select(a => a["display_name"])));                                // tags
                            features.Add(row?["tags"].Count().ToString() ?? "");                                                          // tagCount
                            features.Add(row?["tags"].Select(a => (double)a["followers"]).DefaultIfEmpty().Average().ToString() ?? "");   // tagAvgFollowers
                            features.Add(row?["tags"].Select(a => (double)a["followers"]).DefaultIfEmpty().Sum().ToString() ?? "");       // tagSumFollowers
                            features.Add(row?["tags"].Select(a => (double)a["followers"]).DefaultIfEmpty().Max().ToString() ?? "");       // tagMaxFollowers
                            features.Add(row?["tags"].Select(a => (double)a["total_items"]).DefaultIfEmpty().Average().ToString() ?? ""); // tagAvgTotalItems
                            features.Add(row?["tags"].Select(a => (double)a["total_items"]).DefaultIfEmpty().Sum().ToString() ?? "");     // tagSumTotalItems
                            features.Add(row?["tags"].Select(a => (double)a["total_items"]).DefaultIfEmpty().Max().ToString() ?? "");     // tagMaxTotalItems

                            // Image features
                            features.Add(isAlbum ? (string)row?["images_count"] : "1");                         // imagesCount

                            if (isAlbum)
                            {
                                features.Add((string)row?["images"]?.ElementAtOrDefault(0)?["link"] ?? "");     // img1Uri
                                features.Add(CleanText(row?["images"]?.ElementAtOrDefault(0)?["description"])); // img1Desc
                                features.Add((string)row?["images"]?.ElementAtOrDefault(0)?["type"] ?? "");     // img1Type
                                features.Add(img1FileName);                                                     // img1FileName
                            }
                            else
                            {
                                features.Add((string)row?["link"] ?? "");                                       // img1Uri
                                features.Add(CleanText(row?["title"]));                                         // img1Desc
                                features.Add((string)row?["type"] ?? "");                                       // img1Type
                                features.Add(img1FileName);                                                     // img1FileName
                            }

                            features.Add((string)row?["images"]?.ElementAtOrDefault(1)?["link"] ?? "");         // img2Uri
                            features.Add(CleanText(row?["images"]?.ElementAtOrDefault(1)?["description"]));     // img2Desc
                            features.Add((string)row?["images"]?.ElementAtOrDefault(1)?["type"] ?? "");         // img2Type
                            features.Add(img2FileName);                                                         // img2FileName

                            features.Add((string)row?["images"]?.ElementAtOrDefault(2)?["link"] ?? "");         // img3Uri
                            features.Add(CleanText(row?["images"]?.ElementAtOrDefault(2)?["description"]));     // img3Desc
                            features.Add((string)row?["images"]?.ElementAtOrDefault(2)?["type"] ?? "");         // img3Type
                            features.Add(img3FileName);                                                         // img3FileName

                            // Post features which can leak the label (alt-labels)
                            features.Add((string)row?["datetime"]);                                             // datetime (can leak due to collection dates not being equal)
                            features.Add((string)row?["views"]);                                                // views
                            features.Add((string)row?["ups"]);                                                  // ups
                            features.Add((string)row?["downs"]);                                                // downs
                            features.Add((string)row?["points"]);                                               // points
                            features.Add((string)row?["score"]);                                                // postScore
                            features.Add((string)row?["comment_count"]);                                        // commentCount
                            features.Add((string)row?["favorite_count"]);                                       // favoriteCount

                            //Console.WriteLine(string.Join('\t', features));

                            fs.WriteLine(string.Join('\t', features));
                            rowsWritten++;
                        }

                        Console.WriteLine($"split={split} : Wrote [{rowsWritten}] of [{length}] posts for url {url}\n\n");
                    }
                }

                // Move files into place
                foreach (string fileName in datasetSplits.Where(a => a.Key != "basepath").Select(a => a.Value))
                {
                    File.Move(fileName + ".tmp", fileName, true);
                }

                Console.WriteLine("=============== End of downloading dataset ===============");


                return datasetSplits;
            }
            finally
            {
                mutex.ReleaseMutex();
            }

        }


        private static string DownloadImage(string split, string label, string url)
        {
            if (string.IsNullOrEmpty(url))
            {
                return "";
            }

            var dataDirectoryName = Path.Combine("DataDir", "Images", split, label);
            Directory.CreateDirectory(dataDirectoryName);

            if (url.EndsWith("mp4"))
            {
                url = Path.ChangeExtension(url, "jpg"); // Get jpg frame of mp4
            }

            var uri = new Uri(url);
            var fileName = Path.Combine(dataDirectoryName, Path.GetFileName(uri.AbsolutePath));

            if (!File.Exists(fileName))
            {
                Console.WriteLine($"Downloading {uri} to {fileName}");
                (new WebClient()).DownloadFile(uri, fileName + ".tmp");
                File.Move(fileName + ".tmp", fileName, true);

                if (!CheckImage(fileName))
                {
                    return "";
                }
            }

            return fileName;
        }


        // Validate that images can be loaded/parsed
        private static bool CheckImage(string fileName)
        {
            try {
                // To avoid locking file, use the construct below to load bitmap
                var bytes = File.ReadAllBytes(fileName);
                using (var ms = new MemoryStream(bytes)) {
                    using (Bitmap dst = (Bitmap)Image.FromStream(ms))
                    {
                        // Check for an incorrect pixel format which indicates the loading failed
                        if (dst.PixelFormat == System.Drawing.Imaging.PixelFormat.DontCare)
                        {
                            Console.Error.WriteLine($"Failed to load image {fileName} due to pixel format.");
                            return false;
                        }

                        // Check image size
                        if (dst.Size.IsEmpty || dst.Size.Height <= 0 || dst.Size.Width <= 0)
                        {
                            Console.Error.WriteLine($"Failed to load image {fileName} due to being empty.");
                            return false;
                        }

                        return true;
                    }
                }
            }
            catch (Exception e)
            {
                Console.Error.WriteLine($"Failed to load image {fileName} due to error: {e}.");
                return false;
            }
        }


        // Clean text to make easier to read TSV files (no quoting / newlines)
        private static string CleanText<T>(T inputStr)
        {
            if (inputStr == null)
            {
                return String.Empty; // Replace nulls with empty strings
            }

            string str = inputStr.ToString();

            if (!Regex.IsMatch(str, @"[\""\t\r\n\f]"))
            {
                return str; // No cleaning needed 
            }

            StringBuilder sb = new StringBuilder(str);
            sb.Replace('"', ' ');
            sb.Replace('\t', ' ');
            sb.Replace('\r', ' ');
            sb.Replace('\n', ' ');
            sb.Replace('\f', ' ');

            return sb.ToString();
        }
    }
}