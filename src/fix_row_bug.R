# Code: Fix row bug
# Author: Tyler J Burns, PhD
# Date: October 1, 2022
# Purpose: to output raw data files without unindexed rows
# Notes: R handles unindexed rows by throwing them out. Pandas does not.
library(tidyverse)

# This script is meant to be called in the command line only
args <- commandArgs(trailingOnly = TRUE)
file <- args[1] # user_tweets_scraped.csv
dat <- readr::read_csv(paste0("data/scraped_tweets/", file))

# Weird errors lead to a few NAs and many repeat rows
# The latter seems to come from media objects in very old tweet data (eg. 2010)
dat <- dplyr::filter(dat, !is.na(Tweet))
dat <- dplyr::distinct(dat)

readr::write_csv(dat, "src/tmp.csv")
