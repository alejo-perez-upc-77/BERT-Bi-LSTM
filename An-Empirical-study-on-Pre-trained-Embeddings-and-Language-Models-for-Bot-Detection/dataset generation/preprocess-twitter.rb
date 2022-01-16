# Ruby 2.0
# Reads stdin: ruby preprocess-twitter.rb input.csv output.csv
#
# Script for preprocessing tweets by Romain Paulus
# with small modifications by Jeffrey Pennington  
# CSV processing by Cristian Berrio
# Note: This preprocess script is in based on the Ruby script for preprocessing Twitter data 
# (https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb), used to generate the Twitter GloVe embeddings
# (https://nlp.stanford.edu/projects/glove/)
def tokenize input

	# Different regex parts for smiley faces
	eyes = "[8:=;]"
	nose = "['`\-]?"

	input = input
		.gsub(/https?:\/\/\S+\b|www\.(\w+\.)+\S*/,"<URL>")
		.gsub("/"," / ") # Force splitting words appended with slashes (once we tokenized the URLs, of course)
		.gsub(/@\w+/, "<USER>")
		.gsub(/#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}/i, "<SMILE>")
		.gsub(/#{eyes}#{nose}p+/i, "<LOLFACE>")
		.gsub(/#{eyes}#{nose}\(+|\)+#{nose}#{eyes}/, "<SADFACE>")
		.gsub(/#{eyes}#{nose}[\/|l*]/, "<NEUTRALFACE>")
		.gsub(/<3/,"<HEART>")
		.gsub(/[-+]?[.\d]*[\d]+[:,.\d]*/, "<NUMBER>")
		.gsub(/#\S+/){ |hashtag| # Split hashtags on uppercase letters
			# TODO: also split hashtags with lowercase letters (requires more work to detect splits...)

			hashtag_body = hashtag[1..-1]
			if hashtag_body.upcase == hashtag_body
				result = "<HASHTAG> #{hashtag_body} <ALLCAPS>"
			else
				result = (["<HASHTAG>"] + hashtag_body.split(/(?=[A-Z])/)).join(" ")
			end
			result
		} 
		.gsub(/([!?.]){2,}/){ # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
			"#{$~[1]} <REPEAT>"
		}
		.gsub(/\b(\S*?)(.)\2{2,}\b/){ # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
			# TODO: determine if the end letter should be repeated once or twice (use lexicon/dict)
			$~[1] + $~[2] + " <ELONG>"
		}
		.gsub(/([^a-z0-9()<>'`\-]){2,}/){ |word|
			"#{word.downcase} <ALLCAPS>"
		}

	return input
end

#print tokenize(ARGV[0])

require 'csv'  
first = true
user = ""
CSV.open(ARGV[1], "wb") do |csv_out|
	CSV.foreach(ARGV[0]) do |row|
		if user != row[0]
			puts row[0]
		end
		if first
			csv_out << ["user","text"]
			first = false
		else
			csv_out << [row[0],tokenize(row[1])]
		end
		user = row[0]
	end

end

