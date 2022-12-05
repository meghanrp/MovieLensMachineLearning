#############################################
# Harvard Data Science Capstone - MovieLens
# Name: Meghan Patterson
# Date: 06/23/20
#############################################

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(knitr)
library(gridExtra)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Separate the genres of the edx and validation datasets 
edx <- edx %>% separate_rows(genres, sep="\\|")
validation <- validation %>% separate_rows(genres, sep="\\|")


################################################
# Exploratory data analysis on the edx data set
################################################
summary(edx)
edx %>% as_tibble()

# Count the number of distinct users and movies
edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))
# Count the number of movies per genre
edx %>% count(genres)

# Create a plot to show the distribution of the number of ratings per movie
gg_movie <- edx %>% count(movieId) %>% ggplot(aes(n)) + geom_histogram(bins = 40, color = "darkblue", fill = "pink") + scale_x_log10() + xlab("Number of Movies (log10)") + ylab("Number of Ratings") + ggtitle("Distribution of the Number of Ratings per MovieId")
gg_movie

# Create a plot to show the distribution of the number of ratings inputted per user
gg_user <- edx %>% count(userId) %>% ggplot(aes(n)) + geom_histogram(bins = 40, color = "pink", fill = "darkblue") + scale_x_log10() + xlab("Number of Ratings per User (log10)") + ggtitle("Distribution of the Number of Ratings per UserId")
gg_user

# Create a plot to show the distribution of the ratings for all of the movies
gg_avg_rating_movie <- edx %>% group_by(movieId) %>% summarize(rating = mean(rating)) %>% ggplot(aes(rating)) + geom_histogram(bins = 40, color = "magenta", fill = "palegreen") + xlab("Average Rating per Movie") + ggtitle("Distribution of Average Ratings per Movies")
gg_avg_rating_movie

# Create a plot to show the distribution and average rating inputted per user
gg_avg_rating_user <- edx %>% group_by(userId) %>% summarize(rating = mean(rating)) %>% ggplot(aes(rating)) + geom_histogram(bins = 40, color = "palegreen", fill = "pink") + xlab("Average Rating per User") + ggtitle("Distribution of Average Ratings per User")
gg_avg_rating_user

##############################################
# Create the RMSE function
##############################################
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#######################################################################
# Split the edx dataset into a training set and a test set
# The training set will be used to train the algorithm 
# The test set will be used to test the many algorithms to be created
#######################################################################
set.seed(1234)
split <- createDataPartition(edx$rating, times = 1, p = 0.9, list=FALSE)
training_set <- edx[split,]
test_set <- edx[-split,]
test_set <- test_set %>% semi_join(training_set, by = "movieId") %>% semi_join(training_set, by = "userId")


##############################################################################
# Create the first model - first model assumes rating is same for all movies
##############################################################################
mu <- mean(training_set$rating)
mu

# Calculate RMSE for first model
naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse

# Add results to a table
rmse_results <- data.frame(Model = "Average", RMSE = naive_rmse)
rmse_results


##############################################################################
# Create the second model - second model assumes there is a movie bias
#############################################################################
# Create the movie bias term
mu <- mean(training_set$rating)
movie_bias <- training_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))

# Create a plot to show the distribution of the movie bias
qplot(b_i, data = movie_bias, bins = 10, color = I("white"))

# Calculate the predictions
pred_ratings_movie <- mu + test_set %>% left_join(movie_bias, by = 'movieId') %>% pull(b_i)

#Calculate the RMSE for the second model
movie_bias_results <- RMSE(pred_ratings_movie, test_set$rating)

# Add the results to the preexisting table
rmse_results <- rmse_results %>% add_row(Model = "Movie Bias", RMSE = movie_bias_results)
rmse_results


###################################################################
# Create the third model - third model assumes there is user bias
###################################################################
# Create a plot to show the user bias distribution
training_set %>% group_by(userId) %>% summarize(b_u = mean(rating)) %>% filter(n()>=100) %>% ggplot(aes(b_u)) + geom_histogram(bins=30, color = "black")

# Create the user bias term
user_bias <- training_set %>% 
  left_join(movie_bias, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Calculate the prediction
predicted_ratings_user <- test_set %>% left_join(movie_bias, by = 'movieId') %>% left_join(user_bias, by = 'userId') %>% mutate(pred = mu + b_i+b_u) %>% pull(pred)

# Calculate the RMSE for the third model
user_bias_results <- RMSE(predicted_ratings_user, test_set$rating)

# Add the results to the preexisting table
rmse_results <- rmse_results %>% add_row(Model = "User Bias", RMSE = user_bias_results)
rmse_results


##########################################################
# Fourth model - fourth model assumes there is genre bias
##########################################################
# Create the genre bias term
genre_bias <- training_set %>% left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>%
  group_by(genres) %>% summarize(b_g = mean(rating - mu - b_i - b_u))

# Calculate the prediction
predict_test_with_genre <- test_set %>% left_join(movie_bias, by='movieId') %>% left_join(user_bias, by='userId') %>%
  left_join(genre_bias, by='genres') %>% mutate(pred = mu + b_i + b_u + b_g) %>% pull(pred)

# Calculate the RMSE for the fourth model
genre_model_RMSE_results <- RMSE(predict_test_with_genre, test_set$rating)

# Add the results to the preexisting table
rmse_results <- rmse_results %>% add_row(Model = "Genre Bias", RMSE = genre_model_RMSE_results)
rmse_results


#################################################################################
# Fifth model - fifth model regularizes (adds penalty term) the movie bias model
#################################################################################
# Create the function for the model
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(training_set$rating)
  
  b_i <- training_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  reg_movie_model_predict <- test_set %>%
    left_join(b_i, by = 'movieId') %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  
  return(RMSE(reg_movie_model_predict, test_set$rating))
})

# Create a plot to show the trend of lambdas versus rmses in the model
qplot(lambdas, rmses)

# Calculate the minimum lambda for the model
min_lambda_reg_movie <- lambdas[which.min(rmses)]
min_lambda_reg_movie

# Calculate the minimum RMSE for the model
rmse_reg_movie_model <- min(rmses)
rmse_reg_movie_model

# Add the results to the preexisting table
rmse_results <- rmse_results %>% add_row(Model = "Regularized Movie Bias", RMSE = rmse_reg_movie_model)
rmse_results


###########################################################################################
# Sixth model - sixth model regularizes (adds penalty term) to the movie + user bias model
###########################################################################################
# Create the function for the model
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(training_set$rating)
  
  b_i <- training_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- training_set %>% 
    left_join(b_i, by='movieId') %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))

  reg_user_model_predict <- test_set %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(reg_user_model_predict, test_set$rating))
})

# Create a plot to show the trend of lambdas versus rmses in the model
qplot(lambdas, rmses)

# Calculate the minimum lambda for the model
min_lambda_reg_user <- lambdas[which.min(rmses)]
min_lambda_reg_user

# Calculate the minimum RMSE for the model
rmse_reg_user_model <- min(rmses)
rmse_reg_user_model

# Add the results to the preexisting table
rmse_results <- rmse_results %>% add_row(Model = "Regularized Movie + User Bias", RMSE = rmse_reg_user_model)
rmse_results


#################################################################################################
# Seventh model - seventh model regularizes (adds penalty term) to the movie + user + genre model
#################################################################################################
# Create the function for the model
lambdas <- seq(0, 20, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(training_set$rating)
      
  b_i <- training_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
         
  b_u <- training_set %>% 
    left_join(b_i, by='movieId') %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- training_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - mu - b_u)/(n() + l))
      
  reg_genre_model_predict <- test_set %>%
     left_join(b_i, by = 'movieId') %>%
     left_join(b_u, by='userId') %>%
     left_join(b_g, by='genres') %>%
     mutate(pred = mu + b_i + b_u + b_g) %>%
     pull(pred)
  
    return(RMSE(reg_genre_model_predict, test_set$rating))
})

# Create a plot to show the trend of lambdas versus rmses in the model
qplot(lambdas, rmses)

# Calculate the minimum lambda for the model
min_genre_lambda <- lambdas[which.min(rmses)]
min_genre_lambda

# Calculate the minimum RMSE for the model
rmse_reg_genre_model <- min(rmses)
rmse_reg_genre_model

# Add the results to the preexisting table
rmse_results <- rmse_results %>% add_row(Model = "Regularized Movie + User + Genre Bias", RMSE = rmse_reg_genre_model)
rmse_results


##################################################################################################################################
# Test movie + user + genre model on validation set
# The regularized movie + user + genre model gave the lowest RMSE - so the lambda and model will be used on the validation set
# The training set will be replaced with the edx set in order to test on as much data as possible
##################################################################################################################################
# Min. lambda from the regularized movie + user + genre bias model
lambda <- min_genre_lambda

# Mean rating on the edx dataset
mu_edx <- mean(edx$rating)

# Compute the regularized movie averages with lambda
movie_avgs_reg <- edx %>% group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx) / (n() + lambda))

# Compute the regularized user averages with lambda
user_avg_reg <- edx %>% left_join(movie_avgs_reg, by = "movieId") %>% group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_edx) / (n() + lambda))

# Compute the regularized genre averages with lambda
genre_avg_reg <- edx %>% group_by(genres) %>% left_join(movie_avgs_reg, by = "movieId") %>% left_join(user_avg_reg, by = "userId") %>%
  summarize(b_g = sum(rating - b_i - b_u - mu_edx) / (n() + lambda))

# Compute the prediction
predict_valid <- validation %>% left_join(movie_avgs_reg, by = "movieId") %>%
  left_join(user_avg_reg, by = "userId") %>% left_join(genre_avg_reg, by = "genres") %>%
  mutate(pred = mu_edx + b_i + b_u + b_g) %>% pull(pred)

# Compute the RMSE
valid_model_rmse <- RMSE(predict_valid, validation$rating)

# Add final results to the table
rmse_results <- rmse_results %>% add_row(Model = "Final Regularized Movie + User + Genre Bias", RMSE = valid_model_rmse)
rmse_results
