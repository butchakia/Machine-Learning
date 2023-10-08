#import dataset
animals_training <- read_csv("C:/Users/whitl/OneDrive/Desktop/DSCI508 Machine Learning/Week 5/animals-training.csv")
View(animals_training)

# Creating the data frame (data was having a rough time working when loaded ¯\_(ツ)_/¯ )
data <- data.frame(
  Legs = c(0, 0, 2, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 16, 16),
  Body.Covering = c(
    "scales", "scales", "feathers", "feathers", "feathers", "furry", "furry",
    "furry", "furry", "furry", "hide", "scales", "scales", "scales", "scales",
    "scales", "scales", "scales", "furry", "furry"
  ),
  Animal = c(
    "snake", "snake", "bird", "bird", "bird", "gorilla", "gorilla", "dog", "dog",
    "dog", "cow", "butterfly", "butterfly", "butterfly", "butterfly", "butterfly",
    "butterfly", "butterfly", "caterpillar", "caterpillar"
  )
)


# Split data into train and test
set.seed (85)
library(caret) 

index <- createDataPartition(y = data$Animal,
                             p = .7,
                             list = FALSE)

animals_training.train <- data[index,] 
animals_training.test <- data[-index,]

# Fit a decision tree

library(rpart)
library(rpart.plot)
set.seed(85)

# Convert columns to factors
data$Legs <- as.factor(data$Legs)
data$Body.Covering <- as.factor(data$Body.Covering)
data$Animal <- as.factor(data$Animal)

# Create the formula for the decision tree
target <- "Animal"
features <- c("Legs", "Body.Covering")
dtree.f <- as.formula(paste(target, paste(features, collapse = "+"), sep = "~"))

# Fit the decision tree using the rpart function
dtree <- rpart(dtree.f, data = data, method = "class",
               control = rpart.control(minsplit = 5, cp = 0.01, maxdepth = 5))

# Plot the decision tree
library(rpart.plot)
rpart.plot(dtree)

# Calculate entropy of a vector of class labels
calculate_entropy <- function(labels) {
  class_counts <- table(labels)
  total_samples <- length(labels)
  probabilities <- class_counts / total_samples
  
  # Handle cases where the probabilities are zero
  probabilities[is.nan(probabilities)] <- 0
  
  # Check if there is only one unique class
  if (length(unique(labels)) == 1) {
    entropy <- 0
  } else {
    entropy <- -sum(probabilities * log2(probabilities))
  }
  
  return(entropy)
}

# Calculate entropy for the "Animal" column
entropy_animal <- calculate_entropy(data$Animal)
print("Entropy of the 'Animal' column:")
print(entropy_animal)



# Calculate Gini impurity of a vector of class labels
calculate_gini_impurity <- function(labels) {
  class_counts <- table(labels)
  total_samples <- length(labels)
  probabilities <- class_counts / total_samples
  
  # Handle cases where the probabilities are zero
  probabilities[is.nan(probabilities)] <- 0
  
  gini_impurity <- 1 - sum(probabilities^2)
  return(gini_impurity)
}

# Calculate Gini impurity for the "Animal" column
gini_animal <- calculate_gini_impurity(data$Animal)
print("Gini Impurity of the 'Animal' column:")
print(gini_animal)

# It appears my code thinks legs=6 is the best split point, I know from the data in the modules that is not true.
# Had trouble trying to get the code to use all animals as well. I couldn't really calculate an entropy for my
# model but the Gini was 1 so that's good? Probably not. Didn't have time to calculate the Gini and entropy after
# the optimal split but that would have been tough with these lines of trouble.



