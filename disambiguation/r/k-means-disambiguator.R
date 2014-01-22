################################################################################
## R-script for entity disambiguation clustering.                             ##
##                                                                            ##
## Requires the following parameters to be defined:                           ##
##                                                                            ##
##  n.clusters                                                                ##
##  surface.form                                                              ##
##  WORKING_DIR                                                               ##
##                                                                            ##
################################################################################
# For testing, execute the following line to give default values to parameters.
# n.clusters <- 12; surface.form <- 'John Roberts'; WORKING_DIR <- '/opt/disambiguation/mentions-data';
setwd(WORKING_DIR)
cat(paste("Working directory is: ", WORKING_DIR, "\n", sep=""))
################################################################################
# Define supporting functions.                                                 #
################################################################################
LoadData <- function(contraint="John Roberts") {
  data = read.csv("mentions-matrix.csv", header=F, sep=" ") # nrow(data)
  labels = read.csv("labels.txt", header=F, row.names = NULL, sep="\t")
  names(labels) <- c("index", "doc", "entity", "occurrence")
  labels$doc <- as.character(labels$doc)
  labels$entity <- as.character(labels$entity) # head(labels)
  indices <- (1:nrow(data))[grepl(contraint, labels$entity)]
  rnames <- 1:length(indices)
  data <- data[indices,]
  row.names(data) <- rnames
  labels <- labels[indices,]
  row.names(labels) <- rnames
  list(data, labels)
}
################################################################################
# Load data.                                                                   #
################################################################################
results <- LoadData(surface.form); i <- 1; data <- results[[i]]; i <- i + 1; labels <- results[[i]];
################################################################################
# Cluster data with K-Means                                                    #
################################################################################
fit <- kmeans(data, n.clusters)
cat(paste(n.clusters, " clusters created.  Counts are:", sep=""))
table(fit$cluster)
data.clustered <- labels
data.clustered$cluster <- fit$cluster
indices <- order(fit$cluster)
data.clustered <- data.clustered[indices,]
write.table(data.clustered, file="kmeans-clusters.csv", row.names = F, sep="\t")
