using System;
using System.Collections.Generic;
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

		private static readonly (string split, float weight, string label, string url)[] urls = {
				// Recent user submissions
				("train", 10, "UserSub", "https://api.imgur.com/3/gallery/user/time/7?client_id="),
				("train", 10, "UserSub", "https://api.imgur.com/3/gallery/user/time/6?client_id="),
				("train", 10, "UserSub", "https://api.imgur.com/3/gallery/user/time/5?client_id="),
				("train", 10, "UserSub", "https://api.imgur.com/3/gallery/user/time/4?client_id="),
				("valid", 10, "UserSub", "https://api.imgur.com/3/gallery/user/time/3?client_id="),
				("valid", 10, "UserSub", "https://api.imgur.com/3/gallery/user/time/2?client_id="),
				("test", 10, "UserSub", "https://api.imgur.com/3/gallery/user/time/1?client_id="),
				("test", 10, "UserSub", "https://api.imgur.com/3/gallery/user/time/0?client_id="),

				/*// Random viral images
				("train", 1, "FrontPage", "https://api.imgur.com/3/gallery/random/random/5?client_id="),
				("train", 1, "FrontPage", "https://api.imgur.com/3/gallery/random/random/4?client_id="),
				("train", 1, "FrontPage", "https://api.imgur.com/3/gallery/random/random/3?client_id="),
				("train", 1, "FrontPage", "https://api.imgur.com/3/gallery/random/random/2?client_id="),
				("valid", 1, "FrontPage", "https://api.imgur.com/3/gallery/random/random/1?client_id="),
				("test", 1, "FrontPage", "https://api.imgur.com/3/gallery/random/random/0?client_id="),*/

				// Recent viral images
				("train", 1, "FrontPage", "https://api.imgur.com/3/gallery/hot/time/7?client_id="),
				("train", 1, "FrontPage", "https://api.imgur.com/3/gallery/hot/time/6?client_id="),
				("train", 1, "FrontPage", "https://api.imgur.com/3/gallery/hot/time/5?client_id="),
				("train", 1, "FrontPage", "https://api.imgur.com/3/gallery/hot/time/4?client_id="),
				("valid", 1, "FrontPage", "https://api.imgur.com/3/gallery/hot/time/3?client_id="),
				("valid", 1, "FrontPage", "https://api.imgur.com/3/gallery/hot/time/2?client_id="),
				("test", 1, "FrontPage", "https://api.imgur.com/3/gallery/hot/time/1?client_id="),
				("test", 1, "FrontPage", "https://api.imgur.com/3/gallery/hot/time/0?client_id="),
			};

		public static Dictionary<string, string> GetDataset()
        {
			return GetDataset(urls);
        }

		private static Dictionary<string, string> GetDataset((string split, float weight, string label, string url)[] urls)
		{
			var datasetSplits = new Dictionary<string, string>();

			var dataDirectoryName = "DataDir";
			Directory.CreateDirectory(dataDirectoryName);

			datasetSplits["basepath"] = Path.GetFullPath(".");

			// Create empty datasets w/ a header row
			foreach ((string split, float weight, string label, string url) in urls)
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
				foreach ((string split, float weight, string label, string url) in urls)
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
							"tagAvgTotalItems",
							"tagSumTotalItems",
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
						}));
					}
				}

				// Generate a nonce like "3a22db833f45c0c"
				var r = new Random();
				string clientId = "546c25a59c58ad7"; // (r.Next().ToString("X") + r.Next().ToString("X")).ToLowerInvariant();

				foreach ((string split, float weight, string label, string url) in urls)
				{
					using (var fs = File.AppendText(datasetSplits[split] + ".tmp"))
					{
						Console.WriteLine($"Downloading {url + clientId}");
						string json = new WebClient().DownloadString(url + clientId);
						var jsonDict = JObject.Parse(json);

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

						for (int i = 0; i < jsonDict["data"].Count(); i++)
						{
							var row = jsonDict["data"][i];
							bool isAlbum = (bool)(row?["is_album"]);

							Console.WriteLine($"Time stamp: {DateTimeOffset.FromUnixTimeSeconds((long)row?["datetime"]).DateTime}");

							List<string> features = new List<string>();

							string img1FileName, img2FileName, img3FileName;
							try
							{
								img1FileName = GetImage(split, label, (string)(isAlbum ? row?["images"]?.ElementAtOrDefault(0)?["link"] : row?["link"]));
								img2FileName = GetImage(split, label, (string)(row?["images"]?.ElementAtOrDefault(1)?["link"]));
								img3FileName = GetImage(split, label, (string)(row?["images"]?.ElementAtOrDefault(2)?["link"]));
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
							features.Add(row?["tags"].Select(a => (double)a["total_items"]).DefaultIfEmpty().Average().ToString() ?? ""); // tagAvgTotalItems
							features.Add(row?["tags"].Select(a => (double)a["total_items"]).DefaultIfEmpty().Sum().ToString() ?? "");     // tagSumTotalItems

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

							Console.WriteLine(string.Join('\t', features));

							fs.WriteLine(string.Join('\t', features));
						}
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

		private static string GetImage(string split, string label, string url)
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
				(new WebClient()).DownloadFile(uri, fileName);
			}

			return fileName;
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